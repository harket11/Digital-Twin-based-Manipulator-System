VSCode 설정 안내
VSCode에서 설치할 것:

Python
Python for VSCode
Python Extension Pack
SQLite Viewer
설치 후 반드시 활성화(enable) 해야 합니다.

-------------------------------------------------------------

터미널에서 설치할 패키지:

# 시스템 패키지 업데이트 및 pip 설치
sudo apt update
sudo apt install -y python3-pip

# 필수 Python 패키지 설치
python3 -m pip install ultralytics roboflow

# PyTorch 설치
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# YOLO 관련 패키지 재설치
python3 -m pip install ultralytics

# NumPy 및 pandas 설치/업데이트
python3 -m pip uninstall -y numpy
python3 -m pip install "numpy>=1.26,<2"
python3 -m pip install pandas

# matplotlib 호환 버전 설치
pip uninstall matplotlib matplotlib-inline -y
pip install matplotlib==3.7.2 matplotlib-inline==0.1.6

-----------------------------------------------------------

각 파일 설명 
sub_save.py : cnn 데이터셋 마련을 위한 파일입니다. conveyor.py에서 한 가지 타입의 과일만 나오게 usd 경로를 수정하고, 실행하면 해당 타입 과일의 데이터셋을 얻을 수 있습니다.

train_resnet50_abc.py : 앞서 얻은 데이터셋으로 cnn 파인튜닝을 위한 코드입니다.

best_resnet50_abc.pt : cnn 학습 후 얻은 모델 가중치 파일입니다.

yolov8s.pt : object detection을 위한 yolov8 모델 가중치 파일입니다.

lychee01, pomegranate01 : 사과 클래스 분류를 위한 usd파일을 넣어놓은 폴더입니다.

apple_grade_counts.db : 각 사과를 감지하고 분류 시 업데이트 되는 db파일입니다. 프로그램 실행 시마다 값이 누적 됩니다.

main : 시뮬레이션 실행을 위한 맵 파일과 동작 제어를 위한 파일이 있습니다.

--------------------------------------------------------------

실행 가이드
1. 해당 폴더를 /home/roeky/Desktop/ 경로 아래 위치시킨다.

2. main 폴더 내부의 파일들을 /home/isaacsim/extension_examples/hello_world 폴더에 붙여넣는다.

3. 
cd ~/isaacsim
./post_install.sh
./isaac-sim.sh 실행한다.

4. execute sub_pub.py
다른 터미널을 열어
source /opt/ros/humble/setup.bash
python3 /home/rokey/Desktop/detect_classification/sub_pub.py
를 실행한다.

5.
isaacsim 내부에서 Window - Example - Robotics Examples를 누르고 Robotics Examples - ROKEY - conveyor를 클릭하고, load한다.
