# Telegram MCP Chatbot

Chatbot Telegram tích hợp với Claude AI và Model Context Protocol (MCP) để truy vấn cơ sở dữ liệu MySQL của phòng khám Skin&Beam.

## 🌟 Tính năng chính

- **Tích hợp Claude AI**: Sử dụng Claude Sonnet 4 để xử lý ngôn ngữ tự nhiên
- **Kết nối MySQL**: Truy vấn trực tiếp cơ sở dữ liệu qua SSH tunnel
- **Model Context Protocol (MCP)**: Kiến trúc modular để quản lý kết nối database
- **Hỗ trợ đa ngôn ngữ**: Trả lời bằng tiếng Việt và tiếng Anh
- **Lịch sử hội thoại**: Lưu trữ ngữ cảnh cuộc trò chuyện cho mỗi người dùng
- **Xử lý tin nhắn dài**: Tự động chia nhỏ tin nhắn dài thành nhiều phần

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telegram Bot  │ ── │   Claude API     │ ── │   MySQL DB      │
│                 │    │   (Sonnet 4)     │    │   (via SSH)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
    ┌─────────────────────────────────────────────────────────────┐
    │                MCP MySQL Server                             │
    │  - Schema caching                                           │
    │  - Query optimization                                       │
    │  - Connection pooling                                       │
    └─────────────────────────────────────────────────────────────┘
```

## 📋 Yêu cầu hệ thống

- Python 3.8+
- MySQL database
- SSH access (tùy chọn)
- Telegram Bot Token
- Claude API Key

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd Telegram_MCP_Chatbot
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
Tạo file `.env` với nội dung:

```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Claude API
CLAUDE_API_KEY=your_claude_api_key

# MySQL Database
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_database_name

# SSH Tunnel (tùy chọn)
SSH_HOST=your_ssh_host
SSH_PORT=22
SSH_USER=your_ssh_user
SSH_PASSWORD=your_ssh_password
PRIVATE_KEY=your_private_key_content
```

### 4. Tạo schema database
```bash
python get_schema.py
```

### 5. Chạy bot
```bash
python main.py
```

## 📊 Cấu trúc cơ sở dữ liệu

Bot hỗ trợ truy vấn các bảng chính của hệ thống Skin&Beam:

### Bảng chính
- **customers**: Thông tin khách hàng
- **booking**: Lịch hẹn và trạng thái
- **payment_history**: Lịch sử thanh toán
- **branch_info**: Thông tin chi nhánh
- **employees**: Thông tin nhân viên
- **products**: Danh mục sản phẩm
- **services**: Danh mục dịch vụ

### Mã chi nhánh
- `SB-TSIM-SHA-TSUI` (TST): Chi nhánh Tsim Sha Tsui
- `SB-MONG-KOK` (MK): Chi nhánh Mong Kok  
- `SB-CAUSEWAY-BAY` (CWB): Chi nhánh Causeway Bay

### Trạng thái booking
- `WAITING`: Đang chờ xác nhận
- `CONFIRMED`: Đã xác nhận (chưa đến nếu là hôm nay)
- `CANCELED`: Đã hủy
- `CHECKED_IN`: Đã check-in, đang chờ
- `PAID`: Đang điều trị
- `FINISHED`: Hoàn thành
- `REJECTED`: Từ chối

## 💬 Cách sử dụng

### Lệnh cơ bản
- `/start` - Khởi động bot
- `/clear` - Xóa lịch sử hội thoại

### Ví dụ câu hỏi
```
"Có bao nhiêu khách hàng đặt lịch hôm nay?"
"Doanh thu tháng này của chi nhánh TST?"
"Danh sách bác sĩ đang làm việc"
"Khách hàng nào chưa thanh toán?"
"How many pending appointments today?"
```

## 🔧 Tối ưu hóa hiệu suất

### Query Optimization
- Tự động thêm `LIMIT 10` cho các truy vấn ban đầu
- Sử dụng `COUNT(*)` thay vì select toàn bộ dữ liệu
- Tránh JOIN phức tạp không cần thiết
- Timeout 30 giây cho mỗi truy vấn

### Memory Management
- Cache database schema khi khởi động
- Giới hạn lịch sử hội thoại (20 tin nhắn/user)
- Connection pooling cho MySQL

## 🛠️ Cấu trúc code

```
Telegram_MCP_Chatbot/
├── main.py              # File chính chạy bot
├── get_schema.py        # Script lấy schema database
├── requirements.txt     # Dependencies
├── db_schema.json      # Schema database (auto-generated)
├── backup.txt          # File backup
└── .env                # Biến môi trường (tạo thủ công)
```

### Classes chính

#### `MCPMySQLServer`
- Quản lý kết nối MySQL và SSH tunnel
- Thực thi truy vấn và lấy thông tin schema
- Connection pooling và error handling

#### `ClaudeMCPBot`
- Tích hợp với Claude API
- Xử lý ngữ cảnh hội thoại
- Format và tối ưu hóa response

#### `DecimalEncoder`
- Custom JSON encoder cho Decimal và datetime
- Đảm bảo serialization chính xác

## 🔒 Bảo mật

- SSH tunnel cho kết nối database an toàn
- Environment variables cho sensitive data
- Input validation và SQL injection prevention
- Rate limiting và timeout protection

## 📝 Logging

Bot ghi log các hoạt động quan trọng:
- Kết nối SSH tunnel
- Truy vấn database
- Lỗi xử lý tin nhắn
- Performance metrics

## 🐛 Troubleshooting

### Lỗi thường gặp

**SSH Connection Failed**
```bash
# Kiểm tra SSH credentials và network
ssh -i your_key.pem user@host
```

**MySQL Connection Error**
```bash
# Test MySQL connection
mysql -h host -P port -u user -p database
```

**Claude API Error**
```bash
# Kiểm tra API key và quota
curl -H "x-api-key: your_key" https://api.anthropic.com/v1/messages
```

**Schema Loading Failed**
```bash
# Tạo lại schema file
python get_schema.py
```

## 🚀 Deployment

### Production Setup
1. Sử dụng process manager (PM2, systemd)
2. Setup reverse proxy (nginx)
3. Configure logging rotation
4. Monitor resource usage
5. Setup backup cho database schema

### Docker (tùy chọn)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs
2. Xem phần Troubleshooting
3. Tạo issue trên GitHub
4. Liên hệ team phát triển

---

**Phiên bản**: 1.0.0  
**Cập nhật cuối**: January 2025  
**Tác giả**: Development Team