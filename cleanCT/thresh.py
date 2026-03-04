from pathlib import Path
import time
import cv2

in_dir = 'C:/Users/dumbo/code/BoneProject/micro_CT/micro_CT/micro_CT/S.01/A_Rec/'
out_dir = 'C:/Users/dumbo/code/BoneProject/cleanCT/bmp/'
thresh_val = 85

start = time.time()

directory = Path(in_dir)
for file in directory.iterdir():  
    if not file.is_file() or Path(file).suffix != '.bmp':
        continue
    print(file)
    print(out_dir+Path(file).stem)

    filen = Path(file).stem + '.bmp'
    path = in_dir + filen
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    print(out_dir + filen)
    cv2.imwrite(out_dir + filen, thresh)

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")
