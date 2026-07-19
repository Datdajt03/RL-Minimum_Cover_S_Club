import sys
import os

# Thêm thư mục cha (pcode/ban2) vào python path để import các module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Đổi thư mục làm việc hiện tại về thư mục cha để main.py định vị đúng pbodulieu
os.chdir(parent_dir)

from main import main

if __name__ == '__main__':
    main()
