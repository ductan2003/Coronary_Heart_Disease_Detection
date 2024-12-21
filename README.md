# PointCaré Graph-based Continuous Network

## Giới thiệu
**PointCaré Graph-based Continuous Network (PGCN)** là một phương pháp đề xuất cho bài toán **học liên tục** trong theo dõi **bệnh tim mạch vành** theo **hoạt động hàng ngày**. Phương pháp sử dụng dữ liệu điện tâm đồ kết hợp với hoạt động của người dùng để từ đó phản hồi kết quả dự đoán người dùng có mắc bệnh tim mạch vành hay không. Trong đó:
- PointCaré đại diện cho phương pháp xử lý dữ liệu điện tâm đồ.
- Graph đại diện cho một đồ thị không cố định, được cập nhật liên tục ở tầng biểu diễn.
- Continuous đại diện cho bài toán học liên tục.
- Network đại diện cho kiến trúc mạng được thiết kế nhằm tối ưu hóa việc xử lý và dự đoán.

## Cài đặt
### Sinh dữ liệu điện tâm đồ tự động
Quá trình sinh dữ liệu được thông qua một máy mô phỏng ECG - được giới thiệu bởi McSharry et al - Chi tiết mã nguồn tại [đây](https://www.robots.ox.ac.uk/~gari/CODE/ECGSYN/JAVA/APPLET2/ecgsyn/ecg-java/). Mã nguồn được tuỳ chỉnh tại nhánh `ECG Generator`. Quá trình sinh tự động tuỳ theo các nhãn được thực hiện thông qua file `EcgGen.java`.

### Cài đặt các mô hình
Cài đặt các thư viện liên quan
```sh
pip install -r requirements.txt
```
Di chuyển vào các thư mục mang tên mô hình (KSOM, LLCS, GNG, PGCN) mong muốn và tiến hành huấn luyện
```sh
cd model_name
python model_name.py
```
Lưu ý: Đối với các mô hình LLCS và PGCN, cần chỉnh sửa các tham số tại file `constant.py`

