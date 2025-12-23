import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_project.settings')
django.setup()

from django.contrib.auth.models import User

try:
    user = User.objects.get(username='admin')
    user.set_password('admin123')
    user.save()
    print('✅ Password set successfully for admin user')
    print('Username: admin')
    print('Password: admin123')
except User.DoesNotExist:
    print('❌ Admin user does not exist')
