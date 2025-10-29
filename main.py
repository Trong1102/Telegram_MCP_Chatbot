import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
import aiomysql
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from decimal import Decimal
from datetime import date, datetime

# Only import sshtunnel if SSH is configured
try:
    import sshtunnel
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False
    logging.warning("sshtunnel not available. SSH connections disabled.")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal and date types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

class MCPMySQLServer:
    """MCP Server for MySQL database connection"""
    
    def __init__(self, db_config: Dict, ssh_config: Optional[Dict] = None):
        self.db_config = db_config
        self.ssh_config = ssh_config
        self.tunnel = None
        self.pool = None
        
    async def start(self):
        """Start SSH tunnel and MySQL connection"""
        if self.ssh_config and SSH_AVAILABLE:
            try:
                # Prepare SSH parameters
                ssh_params = {
                    'ssh_username': self.ssh_config['username'],
                    'remote_bind_address': (self.db_config['host'], self.db_config['port'])
                }
                
                # Use key file if provided, otherwise use password
                # if self.ssh_config.get('key_file'):
                #     ssh_params['ssh_pkey'] = self.ssh_config['key_file']
                # elif self.ssh_config.get('password'):
                #     ssh_params['ssh_password'] = self.ssh_config['password']
                if self.ssh_config.get('private_key'):
                    # Tạo file tạm cho private key
                    key_content = self.ssh_config['private_key']
                    key_path = "temp_key.pem"
                    with open(key_path, "w") as key_file:
                        key_file.write(key_content)
                    os.chmod(key_path, 0o600)  # quyền đọc riêng tư
                    ssh_params['ssh_pkey'] = key_path
                elif self.ssh_config.get('password'):
                    ssh_params['ssh_password'] = self.ssh_config['password']
                
                self.tunnel = sshtunnel.SSHTunnelForwarder(
                    (self.ssh_config['host'], self.ssh_config.get('port', 22)),
                    **ssh_params
                )
                self.tunnel.start()
                
                # Wait for tunnel to be ready
                import time
                time.sleep(2)
                
                # Update db_config to use tunnel
                self.db_config['host'] = '127.0.0.1'
                self.db_config['port'] = self.tunnel.local_bind_port
                
                logger.info(f"SSH tunnel established on local port {self.tunnel.local_bind_port}")
            except Exception as e:
                logger.error(f"Failed to create SSH tunnel: {str(e)}")
                logger.info("Attempting direct connection without SSH tunnel...")
        elif self.ssh_config and not SSH_AVAILABLE:
            logger.warning("SSH configuration provided but sshtunnel not available. Using direct connection.")
            
        # Create MySQL connection pool
        self.pool = await aiomysql.create_pool(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            db=self.db_config['database'],
            minsize=1,
            maxsize=10
        )
        
    async def stop(self):
        """Stop MySQL connection and SSH tunnel"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            
        if self.tunnel:
            self.tunnel.stop()
            
    async def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query)
                result = await cursor.fetchall()
                return result
                
    async def get_table_info(self, table_name: str) -> Dict:
        """Get table structure information"""
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{self.db_config['database']}' 
        AND TABLE_NAME = '{table_name}'
        """
        columns = await self.execute_query(query)
        return {
            'table_name': table_name,
            'columns': columns
        }
        
    async def get_database_schema(self) -> List[Dict]:
        """Get all tables and their structures"""
        # Get all tables
        tables_query = f"""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{self.db_config['database']}'
        """
        tables = await self.execute_query(tables_query)
        
        schema_info = []
        for table in tables:
            table_info = await self.get_table_info(table['TABLE_NAME'])
            schema_info.append(table_info)
            
        return schema_info

class ClaudeMCPBot:
    """Telegram bot with Claude API and MCP integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.anthropic = AsyncAnthropic(api_key=config['claude_api_key'])
        self.mcp_server = MCPMySQLServer(
            config['mysql_config'],
            config.get('ssh_config')
        )
        self.context_cache = {}
        self.conversation_history = {}  # Store conversation history per user
        self.db_schema = None  # Will be set during initialization
        
#     async def start_mcp(self):
#         """Initialize MCP server and fetch database schema once"""
#         await self.mcp_server.start()
#         logger.info("MCP MySQL server started")
        
#         # Fetch and store database schema once at startup
#         logger.info("Fetching database schema...")
#         try:
#             schema = await self.mcp_server.get_database_schema()
            
#             # Format schema for context
#             context = "Database Schema:\n"
#             for table in schema:
#                 context += f"\nTable: {table['table_name']}\n"
#                 context += "Columns:\n"
#                 for col in table['columns']:
#                     context += f"  - {col['COLUMN_NAME']} ({col['DATA_TYPE']})"
#                     if col['COLUMN_KEY'] == 'PRI':
#                         context += " PRIMARY KEY"
#                     if col['IS_NULLABLE'] == 'NO':
#                         context += " NOT NULL"
#                     context += "\n"
            
#             self.db_schema = context
#             logger.info(f"Database schema loaded successfully with {len(schema)} tables")
            
#             # Optionally save schema to file
#             try:
#                 with open('db_schema.json', 'w', encoding='utf-8') as f:
#                     json.dump(schema, f, ensure_ascii=False, indent=2)
#                 logger.info("Schema saved to db_schema.json")
#             except Exception as e:
#                 logger.warning(f"Could not save schema to file: {e}")
            
#         except Exception as e:
#             logger.error(f"Failed to load database schema: {str(e)}")
#             # Set minimal schema as fallback
#             self.db_schema = """Database tables available:
# - customers: Customer information
# - branches: Clinic branches
# - booking: Appointment bookings
# - payment_history: Payment records
# - products: Product catalog
# - services: Service catalog
# - employees: Staff information
# - customer_ranks: Customer ranking system"""
    async def start_mcp(self):
        """Initialize MCP server and fetch database schema once"""
        await self.mcp_server.start()
        logger.info("MCP MySQL server started")
        
        try:
            # Sử dụng file schema đã tối ưu
            schema_file = 'db_schema_optimized.json'  # Thay đổi từ db_schema.json
            
            if os.path.exists(schema_file):
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
            
            # Format schema ngắn gọn hơn
            context = "Database Schema:\n\n"
            for table in schema_data:
                context += f"{table['table_name']}: "
                cols = [col['COLUMN_NAME'] for col in table['columns']]
                context += f"{', '.join(cols)}\n"
            
            self.db_schema = context
            logger.info(f"Schema loaded: {len(context)} characters (optimized)")
        except Exception as e:
            logger.error(f"Failed to load database schema: {str(e)}")
            # Set minimal schema as fallback
            self.db_schema = """Database tables available:
- customers: Customer information
- branches: Clinic branches
- booking: Appointment bookings
- payment_history: Payment records
- products: Product catalog
- services: Service catalog
- employees: Staff information
- customer_ranks: Customer ranking system"""
    async def stop_mcp(self):
        """Stop MCP server"""
        await self.mcp_server.stop()
        logger.info("MCP MySQL server stopped")
        
    async def get_database_context(self) -> str:
        """Get database schema context for Claude - no fetching needed"""
        if self.db_schema:
            return self.db_schema
        else:
            # Fallback if schema not loaded
            return "Database schema not available. Please restart the bot."
        
    async def process_database_query(self, user_query: str, user_id: int = None) -> str:
        """Process user query about database"""
        try:
            # Set shorter timeout and use more efficient approach
            async with asyncio.timeout(30):  # Reduced from 25 seconds
                # Get database context
                db_context = await self.get_database_context()
                
                # Get conversation history for this user
                messages = []
                if user_id and user_id in self.conversation_history:
                    messages = self.conversation_history[user_id][-10:]  # Last 10 messages
                
                # Add current user message
                messages.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Create prompt for Claude
                system_prompt = f"""You are a helpful assistant that answers questions about a clinic database.
                
                {db_context}
                
                CRITICAL PERFORMANCE RULES:
                1. ALWAYS use LIMIT 10 for initial queries
                2. Use COUNT(*) instead of selecting all rows when counting
                3. Avoid complex JOINs unless necessary
                4. For status checks, use simple WHERE clauses
                
                Example efficient queries:
                - Count pending bookings: SELECT COUNT(*) FROM booking WHERE status = 'pending'
                - Recent bookings: SELECT * FROM booking ORDER BY created_at DESC LIMIT 10
                - Customer count: SELECT COUNT(DISTINCT customer_id) FROM booking WHERE status = 'pending'
                
                IMPORTANT:
                - Unit: HKD
                - Generate SQL in <sql></sql> tags
                - Respond in the language requested by user
                - If query might be slow, use COUNT or LIMIT first
                - For "chưa được làm" (not completed), look for status != 'completed' or similar"""
                
                # Get response from Claude
                message = await self.anthropic.messages.create(
                    model="claude-sonnet-4-20250514",  # Claude Sonnet 4
                    max_tokens=4096,
                    temperature=0,
                    system=system_prompt,
                    messages=messages
                )
                
                response_text = message.content[0].text
                
                # Store conversation history
                if user_id:
                    if user_id not in self.conversation_history:
                        self.conversation_history[user_id] = []
                    self.conversation_history[user_id].append({
                        "role": "assistant",
                        "content": response_text
                    })
                    # Keep only last 20 messages
                    if len(self.conversation_history[user_id]) > 20:
                        self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
                
                # Extract and execute SQL if present
                import re
                sql_pattern = r'<sql>(.*?)</sql>'
                sql_matches = re.findall(sql_pattern, response_text, re.DOTALL)
                
                if sql_matches:
                    # Execute queries and get results
                    all_results = []
                    for sql in sql_matches:
                        sql = sql.strip()
                        try:
                            results = await self.mcp_server.execute_query(sql)
                            if results:
                                all_results.extend(results)
                        except Exception as e:
                            logger.error(f"Error executing query: {str(e)}")
                    
                    # Remove SQL tags from response
                    response_text = re.sub(sql_pattern, '', response_text, flags=re.DOTALL).strip()
                    
                    # If we have results, format them nicely
                    if all_results:
                        # For COUNT queries, just use the count directly
                        if len(all_results) == 1 and 'COUNT(*)' in str(all_results[0]):
                            count_result = list(all_results[0].values())[0]
                            if "tiếng anh" in user_query.lower() or "english" in user_query.lower():
                                response_text = f"There are currently {count_result} customers with pending appointments that haven't been completed yet."
                            else:
                                response_text = f"Hiện tại có {count_result} khách hàng đặt lịch mà chưa được làm."
                        else:
                            # For other queries, format nicely
                            results_json = json.dumps(all_results[:10], ensure_ascii=False, cls=DecimalEncoder)
                            
                            format_prompt = f"""Based on these query results, provide a clear, natural response that answers: "{user_query}"

Results: {results_json}

Format the response nicely without showing raw data or SQL."""
                            
                            format_message = await self.anthropic.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=1000,
                                temperature=0,
                                system="Format database results clearly. Respond in the language requested by the user.",
                                messages=[{"role": "user", "content": format_prompt}]
                            )
                            
                            response_text = format_message.content[0].text
                
                return response_text
                
        except asyncio.TimeoutError:
            return "⏱️ Xử lý mất quá nhiều thời gian. Vui lòng thử lại với câu hỏi đơn giản hơn."
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"❌ Xin lỗi, đã xảy ra lỗi: {str(e)}"

# Telegram bot handlers
async def send_long_message(message, text, parse_mode='Markdown', max_length=4000):
    """Send long messages by splitting them into chunks"""
    # Split message into chunks
    chunks = []
    
    # Try to split by code blocks first
    if '```' in text:
        parts = text.split('```')
        current_chunk = ""
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Normal text
                current_chunk += part
            else:  # Code block
                code_block = f"```{part}```"
                if len(current_chunk) + len(code_block) > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                    chunks.append(code_block[:max_length])
                    current_chunk = ""
                else:
                    current_chunk += code_block
                    
        if current_chunk:
            chunks.append(current_chunk)
    else:
        # Simple split by length
        for i in range(0, len(text), max_length):
            chunks.append(text[i:i + max_length])
    
    # Send each chunk
    for i, chunk in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(0.5)  # Small delay between messages
            
        try:
            await message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            # If markdown fails, try plain text
            await message.reply_text(chunk)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "Xin chào! Tôi là chatbot hỗ trợ trả lời câu hỏi về database Skin&Beam clinic.\n"
        "Bạn có thể hỏi tôi về:\n"
        "- Thông tin bệnh nhân\n"
        "- Lịch hẹn khám\n"
        "- Thông tin bác sĩ\n"
        "- Chi nhánh phòng khám\n"
        "- Doanh thu và thanh toán\n"
        "- Và các thông tin khác trong database\n\n"
        "Commands:\n"
        "/start - Bắt đầu\n"
        "/clear - Xóa lịch sử chat\n\n"
        "Hãy đặt câu hỏi của bạn!"
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command to clear conversation history"""
    user_id = update.effective_user.id
    bot = context.bot_data.get('claude_bot')
    
    if bot and user_id in bot.conversation_history:
        del bot.conversation_history[user_id]
        await update.message.reply_text("✅ Lịch sử trò chuyện đã được xóa!")
    else:
        await update.message.reply_text("Không có lịch sử trò chuyện để xóa.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages"""
    user_message = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Send immediate response to avoid timeout
    processing_msg = await update.message.reply_text("🔍 Đang xử lý câu hỏi của bạn...")
    
    # Process query
    bot = context.bot_data.get('claude_bot')
    if bot:
        try:
            response = await bot.process_database_query(user_message, user_id)
            
            # Delete processing message
            await processing_msg.delete()
            
            # Check if response is too long
            if len(response) > 4000:
                await send_long_message(update.message, response)
            else:
                # Try to send with Markdown first, fallback to plain text if error
                try:
                    await update.message.reply_text(response, parse_mode='Markdown')
                except Exception as e:
                    # If Markdown parsing fails, send as plain text
                    await update.message.reply_text(response)
        except asyncio.TimeoutError:
            await processing_msg.edit_text("⏱️ Processing took too long. Please try again with a simpler question.")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await processing_msg.edit_text("❌ Có lỗi xảy ra. Vui lòng thử lại sau.")
    else:
        await update.message.reply_text("Bot chưa được khởi tạo. Vui lòng thử lại sau.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    
    # Only try to send error message if we have a valid chat
    if update and update.effective_chat:
        try:
            # Use a short timeout to avoid cascading timeouts
            await asyncio.wait_for(
                context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ An error occurred. Please try again with a simpler question."
                ),
                timeout=5.0
            )
        except Exception as e:
            # If we can't even send the error message, just log it
            logger.error(f"Failed to send error message: {e}")

async def post_init(application: Application):
    """Initialize bot after application starts"""
    print("DEBUG - CLAUDE_API_KEY:", os.getenv("CLAUDE_API_KEY"))
    config = {
        'claude_api_key': os.getenv('CLAUDE_API_KEY'),
        'mysql_config': {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DATABASE')
        }
    }
    
    # Add SSH config if provided
    if os.getenv('SSH_HOST'):
        config['ssh_config'] = {
            'host': os.getenv('SSH_HOST'),
            'port': int(os.getenv('SSH_PORT', 22)),
            'username': os.getenv('SSH_USER'),
            'password': os.getenv('SSH_PASSWORD'),
            'private_key': os.getenv('PRIVATE_KEY')
            # 'key_file': os.getenv('SSH_KEY_FILE')
        }
    
    bot = ClaudeMCPBot(config)
    await bot.start_mcp()
    application.bot_data['claude_bot'] = bot

async def post_shutdown(application: Application):
    """Cleanup when application shuts down"""
    bot = application.bot_data.get('claude_bot')
    if bot:
        await bot.stop_mcp()

def main():
    """Main function"""
    # Get Telegram bot token
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_token:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN environment variable")
    
    # Create application
    application = Application.builder().token(telegram_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Add post init and shutdown
    application.post_init = post_init
    application.post_shutdown = post_shutdown
    
    # Run bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()