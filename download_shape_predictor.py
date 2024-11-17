import bz2
import requests
import os

def download_landmarks_model():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    save_path = "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_path = "shape_predictor_68_face_landmarks.dat"
    
    # Remove existing files if they exist
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(extracted_path):
        os.remove(extracted_path)
    
    print("Downloading shape predictor file...")
    response = requests.get(url, allow_redirects=True)
    
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print("Download completed!")
    
    print("Extracting file...")
    with bz2.BZ2File(save_path) as fr, open(extracted_path, 'wb') as fw:
        fw.write(fr.read())
    print("Extraction completed!")
    
    # Remove the compressed file
    os.remove(save_path)
    print("Cleaned up compressed file")
    print(f"Shape predictor file is ready at: {extracted_path}")

if __name__ == "__main__":
    download_landmarks_model()
