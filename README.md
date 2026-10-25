# Nhóm 3 - Large-Scale Community Detection via Advanced Graph Partitioning

Phát hiện cộng đồng trong mạng xã hội quy mô lớn bằng thuật toán Multilevel kết hợp song song hóa.

## Thành viên nhóm

| Họ tên | Khóa | Nhiệm vụ |
|---|---|---|
| Bế Việt Hùng | 65 | Code + thực nghiệm |
| Vũ Hoàng An | 63 | Mô hình hóa bài toán |
| Trịnh Việt Thông | 64 | Cơ sở lý thuyết |
| Bùi Văn Dũng | 65 | Công trình liên quan |
| Bùi Ngọc Linh | 66 | Tổng hợp báo cáo |
| Nguyễn Tuấn Hiệp | 66 | Slide + thuyết trình |

## Tập dữ liệu

Tải từ Stanford SNAP Database:

- **Facebook**: https://snap.stanford.edu/data/ego-Facebook.html
- **Youtube**: https://snap.stanford.edu/data/com-Youtube.html

Sau khi tải, đặt vào thư mục `data/`:

data/
├── facebook_combined.txt
└── com-youtube.ungraph.txt


## Cài đặt

```bash
git clone https://github.com/VietHung48/Nhom3_Large-ScaleCommunityDetectionviaAdvancedGraphPartitioning.git
cd Nhom3_Large-ScaleCommunityDetectionviaAdvancedGraphPartitioning
pip install -r requirements.txt
```

## Cấu trúc thư mục

├── data/ # Dữ liệu đồ thị
├── src/
│ ├── utils.py # Đọc dữ liệu, thống kê
│ ├── baseline.py # Louvain gốc
│ ├── coarsen.py # Thu nhỏ đồ thị
│ ├── refine.py # Tinh chỉnh phân cụm
│ ├── multilevel.py # Pipeline đầy đủ: thu nhỏ + phân cụm + phóng to
│ └── parallel.py # Song song hóa bằng multiprocessing
├── third_party/
│ └── python-louvain-master/ # Thư viện Louvain
├── results/ # Kết quả thực nghiệm, thu lại bằng file csv
├── requirements.txt
└── README.md

## Chạy thực nghiệm

Mặc định các file chạy trên dataset Youtube. 
Nếu muốn chạy trên Facebook, mở từng file trong `src/` và sửa dòng:

```python
G = load_youtube()   # đổi thành load_facebook() nếu muốn chạy Facebook
```

### 1. Kiểm tra dữ liệu
Mở `src/utils.py` và bấm Run.

### 2. Chạy thuật toán tham chiếu (Louvain gốc)
Mở `src/baseline.py` và bấm Run.

### 3. Chạy Multilevel không tinh chỉnh
Mở `src/multilevel.py` và bấm Run.

### 4. Chạy Multilevel có tinh chỉnh + song song
Mở `src/parallel.py` và bấm Run.

Kết quả lưu tự động vào `results/`.

## Kết quả

### Facebook (4.039 đỉnh)

| Phương pháp | Q | Thời gian |
|---|---|---|
| Louvain gốc | 0,8349 | 0,998s |
| Multilevel chưa tinh chỉnh | 0,8028 | 0,427s |
| Multilevel có tinh chỉnh (k=1) | 0,8312 | 6,102s |
| Song song k=2 | 0,8204 | 5,388s |
| Song song k=4 | 0,8109 | 6,656s |

### Youtube (1.134.890 đỉnh)

| Phương pháp | Q | Thời gian |
|---|---|---|
| Louvain gốc | 0,7161 | 405s |
| Multilevel chưa tinh chỉnh | 0,6719 | 1030s |
| Multilevel có tinh chỉnh (k=1) | 0,7243 | 2701s |
| Song song k=2 | 0,7149 | 2808s |
| Song song k=4 | 0,6948 | 2186s |

## Lưu ý

- Yêu cầu RAM tối thiểu 8GB cho Youtube. Trên Windows với RAM 8GB, song song hóa chạy tuần tự từng vùng do giới hạn bộ nhớ.
- Facebook dùng để kiểm chứng thuật toán, Youtube là tập dữ liệu chính.

## Tài liệu tham khảo

[1] Blondel et al. (2008). Fast unfolding of communities in large networks. https://arxiv.org/abs/0803.0476

[2] Karypis & Kumar (1998). A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs. https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf

[3] Traag et al. (2019). From Louvain to Leiden. https://www.nature.com/articles/s41598-019-41695-z

[4] Taynaud (2023). python-louvain. https://github.com/taynaud/python-louvain

[5] Stanford SNAP - Facebook. https://snap.stanford.edu/data/ego-Facebook.html

[6] Stanford SNAP - Youtube. https://snap.stanford.edu/data/com-Youtube.html
