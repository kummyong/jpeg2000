import numpy as np
import pandas as pd
from PIL import Image
import os

def create_sample_csv(filepath, rows, cols):
    """-65500 ~ +65500 범위의 int32 CSV 파일을 생성합니다."""
    print(f"'{filepath}' 경로에 int32 범위의 예제 CSV 파일을 생성합니다.")
    data = np.random.randint(-65500, 65501, size=(rows, cols), dtype=np.int32)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, header=False)
    print("CSV 파일 생성 완료.")

def compress_csv_to_jp2_pillow(csv_path, jp2_path, compression_ratio):
    """int32 CSV를 정규화하여 uint16으로 변환 후 압축합니다."""
    print(f"'{csv_path}' 파일을 읽어옵니다.")
    original_data = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.int32)
    
    min_val = np.min(original_data)
    max_val = np.max(original_data)
    data_range = max_val - min_val

    print(f"데이터를 uint16 범위로 정규화합니다 (원본 범위: {min_val}~{max_val}).")
    normalized_data = ((original_data - min_val) / data_range * 65535).astype(np.uint16)

    print(f"데이터를 '{jp2_path}' 파일로 압축합니다. (압축 목표: 1/{compression_ratio})")
    image = Image.fromarray(normalized_data, mode='I;16')
    
    target_bpp = 16 / compression_ratio
    
    image.save(jp2_path, 'JPEG2000', quality_mode='rates', quality_layers=[target_bpp])
    
    original_size = os.path.getsize(csv_path)
    compressed_size = os.path.getsize(jp2_path)
    
    print("압축 완료.")
    print(f"  - 원본 CSV 크기: {original_size / 1024:.2f} KB")
    print(f"  - 압축 JP2 크기: {compressed_size / 1024:.2f} KB")
    
    return original_data, min_val, max_val

def decompress_jp2_to_data_pillow(jp2_path, min_val, max_val):
    """압축된 파일을 읽어 원본 int32 범위로 역정규화합니다."""
    print(f"'{jp2_path}' 파일로부터 데이터를 복원합니다.")
    with Image.open(jp2_path) as image:
        restored_normalized_data = np.array(image, dtype=np.uint16)
    
    print(f"데이터를 원본 범위로 역정규화합니다 ({min_val}~{max_val}).")
    data_range = max_val - min_val
    restored_data = ((restored_normalized_data / 65535) * data_range + min_val)
    restored_data = np.round(restored_data).astype(np.int32)

    print("복원 완료.")
    return restored_data

def save_restored_to_csv(filepath, data):
    """복원된 데이터를 CSV 파일로 저장합니다."""
    print(f"복원된 데이터를 '{filepath}' 파일로 저장합니다.")
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, header=False)
    print("CSV 저장 완료.")

def calculate_and_print_error(original, restored):
    print("원본과 복원된 데이터 간의 오차를 계산합니다.")
    if original.shape != restored.shape:
        print("오류: 데이터의 차원이 일치하지 않아 오차를 계산할 수 없습니다.")
        return

    data_range = np.max(original) - np.min(original)

    abs_error = np.abs(original.astype(np.float64) - restored.astype(np.float64))
    
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))
    max_error = np.max(abs_error)
    
    mae_percentage = (mae / data_range) * 100

    print(f"  - 평균 절대 오차 (MAE): {mae:.4f}")
    print(f"  - 평균 제곱근 오차 (RMSE): {rmse:.4f}")
    print(f"  - 최대 절대 오차: {max_error:.0f}")
    print("-" * 20)
    print(f"  - 평균 절대 오차율: {mae_percentage:.4f} %")
    print(f"    (데이터 실제 범위 {data_range} 대비 평균 오차의 비율)")

def main():
    csv_filepath = 'sample_data_int32.csv'
    jp2_filepath = 'compressed_data_int32.jp2'
    restored_csv_filepath = 'restored_data.csv' # 복원된 CSV 파일 경로
    COMPRESSION_RATIO = 5
    
    create_sample_csv(csv_filepath, 512, 512)
    
    print("\n" + "="*30 + "\n")
    
    original_data, min_val, max_val = compress_csv_to_jp2_pillow(csv_filepath, jp2_filepath, COMPRESSION_RATIO)
    
    print("\n" + "="*30 + "\n")
    
    restored_data = decompress_jp2_to_data_pillow(jp2_filepath, min_val, max_val)
    
    # 복원된 데이터를 CSV로 저장
    save_restored_to_csv(restored_csv_filepath, restored_data)

    print("\n" + "="*30 + "\n")
    
    calculate_and_print_error(original_data, restored_data)

if __name__ == '__main__':
    main()