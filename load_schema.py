import json
import os

class SchemaPersistence:
    """Helper to save and load database schema from file"""
    
    @staticmethod
    async def save_schema(bot_instance, filename='db_schema.json'):
        """Save current schema to file"""
        try:
            schema = await bot_instance.mcp_server.get_database_schema()
            
            # Save raw schema data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Schema saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving schema: {str(e)}")
            return False
    
    @staticmethod
    def load_schema(filename='db_schema.json'):
        """Load schema from file"""
        try:
            if not os.path.exists(filename):
                return None
                
            with open(filename, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                
            # Format schema for context
            context = "Database Schema (from file):\n"
            for table in schema:
                context += f"\nTable: {table['table_name']}\n"
                context += "Columns:\n"
                for col in table['columns']:
                    context += f"  - {col['COLUMN_NAME']} ({col['DATA_TYPE']})"
                    if col['COLUMN_KEY'] == 'PRI':
                        context += " PRIMARY KEY"
                    if col['IS_NULLABLE'] == 'NO':
                        context += " NOT NULL"
                    context += "\n"
                    
            return context
            
        except Exception as e:
            print(f"❌ Error loading schema: {str(e)}")
            return None

# Usage in your bot:
# 1. To save schema once (run this separately):
"""
async def save_schema_once():
    config = {...}  # your config
    bot = ClaudeMCPBot(config)
    await bot.start_mcp()
    await SchemaPersistence.save_schema(bot)
    await bot.stop_mcp()
    
# Run: asyncio.run(save_schema_once())
"""

# 2. In your bot initialization, try to load from file first:
"""
async def start_mcp(self):
    await self.mcp_server.start()
    logger.info("MCP MySQL server started")
    
    # Try to load schema from file first
    self.db_schema = SchemaPersistence.load_schema()
    
    if self.db_schema:
        logger.info("Database schema loaded from file")
    else:
        # Fetch from database if file not found
        logger.info("Fetching database schema from database...")
        # ... existing code to fetch schema ...
"""