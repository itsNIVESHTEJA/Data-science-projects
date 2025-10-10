ExcelECGApp (Admin Approval + Monitor)

Features:
- Excel auth with roles (Admin/User), email storage
- Registration requests -> Admin approval (pending_users.xlsx)
- Admin dashboard: Pending users, Login monitor
- User and Admin: Home, Classification, Description, Result pages
- ECG prediction route using ecg_cnn_cpu_model.h5 (place at project root)

Quickstart:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

First run seeds a default admin if none exists:
  username: admin
  password: admin123

Change it immediately after login.

Run:
python app.py
