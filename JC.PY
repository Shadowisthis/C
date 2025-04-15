import sys
import os
import time
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 配置日志记录器
from logging.handlers import TimedRotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler(
            'system.log',
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ],
    encoding='utf-8'
)

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    update_count = pyqtSignal(int, int, float)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    model_loaded = pyqtSignal(bool)
    
    # 添加参数更新信号
    params_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.params = {
            'track_dist': 50,
            'detect_bottom': 20,
            'min_area': 50,
            'max_area': 5000,
            'model_conf': 0.5,
            'model_nms': 0.4,
            'model_size': 640,
            'track_enabled': True,  # 启用跟踪
            'track_buffer': 30,     # 跟踪缓冲区大小
            'track_min_hits': 3     # 最小匹配次数
        }
        self.model = None
        self.model_path = ""
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.source = 0
        self.running = False
        self.pause = False
        self.cap = None
        self.tracked_objects = {}
        self.next_id = 1
        self.frame_height = 0
        self.total_count = 0
        self.counted_ids = set()
        self.device = self._get_optimal_device()
        # 获取可用摄像头列表
        self.available_cameras = self._get_available_cameras()
        logging.info(f"可用摄像头: {self.available_cameras}")
        # 启动线程
        self.start()

    def _get_optimal_device(self):
        """获取最优的计算设备"""
        # 检查是否为树莓派环境
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    # 检查VideoCore GPU是否可用
                    if os.path.exists('/dev/vchiq'):
                        return 'gpu'  # 使用树莓派GPU
        except:
            pass  # 非树莓派环境，继续检查其他设备

        # 检查其他计算设备
        if torch.cuda.is_available():
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        elif hasattr(torch, 'vulkan') and torch.vulkan.is_available():
            return 'vulkan'
        elif hasattr(torch, 'opencl') and torch.opencl.is_available():
            return 'opencl'
        return 'cpu'

    def _get_available_cameras(self):
        """获取系统中可用的摄像头列表"""
        available_cameras = {}
        for i in range(10):  # 检查前10个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # 获取摄像头信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    available_cameras[i] = {
                        'index': i,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    }
                cap.release()
        return available_cameras

    def reset_counters(self):
        with QMutexLocker(self.mutex):
            self.total_count = 0
            self.counted_ids.clear()
            self.tracked_objects = {}
            self.next_id = 1

    def load_model(self, model_path):
        try:
            logging.info(f"开始加载模型: {model_path}")
            if self.model:
                logging.info("清理现有模型资源")
                del self.model
                self.model = None
                torch.cuda.empty_cache()

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            if not model_path.lower().endswith(('.pt', '.pth')):
                raise ValueError("仅支持PyTorch格式模型")
            
            self.model = YOLO(model_path).to(self.device)
            self.model.fuse()
            self.params['model_size'] = self.model.args.get('imgsz', 640)
            self.model_path = model_path
            logging.info(f"模型加载成功，使用设备: {self.device}")
            self.model_loaded.emit(True)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)
            self.model_loaded.emit(False)

    def set_source(self, source):
        try:
            logging.info(f"设置视频源: {source}")
            if self.cap and self.cap.isOpened():
                logging.info("释放现有视频源")
                self.cap.release()
            
            self.source = source
            # 检查是否为摄像头源
            if isinstance(source, int):
                if source not in self.available_cameras:
                    raise ValueError(f"摄像头 {source} 不可用")
                logging.info(f"使用摄像头 {source}: {self.available_cameras[source]}")
            else:
                # 检查视频文件是否存在
                if not os.path.exists(source):
                    raise FileNotFoundError(f"视频文件不存在: {source}")
                if not source.lower().endswith(('.mp4', '.avi', '.mkv')):
                    raise ValueError("不支持的视频格式")
            
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频源: {source}")
            
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"视频源打开成功，帧高度: {self.frame_height}")
            self.reset_counters()
            logging.info("计数器已重置")
        except Exception as e:
            error_msg = f"设置视频源失败: {str(e)}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)

    def run(self):
        while True:
            try:
                self.mutex.lock()
                if not self.running:
                    self.condition.wait(self.mutex)
                self.mutex.unlock()
                
                if self.pause or not self.cap or not self.cap.isOpened():
                    self.msleep(50)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    if self.source != 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.finished_signal.emit()
                    continue
                
                start_time = time.time()
                processed_frame, current_count = self.process_frame(frame)
                self.total_count += current_count
                elapsed = time.time() - start_time

                if processed_frame is not None:
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    convert_to_qt_format = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                    self.change_pixmap.emit(convert_to_qt_format)
                    self.update_count.emit(self.total_count, current_count, elapsed)

            except Exception as e:
                self.error_signal.emit(str(e))
            
            self.msleep(20)

    def process_frame(self, frame):
        results = self.detect_objects(frame)
        current_count = 0
        logging.debug(f"检测到 {len(results)} 个目标")

        for (box, track_id) in results:
            x, y, w, h = [int(v) for v in box]
            
            # 绘制检测框和跟踪ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x, y-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            # 触底检测
            if (y + h) > (self.frame_height - self.params['detect_bottom']):
                if track_id != -1 and track_id not in self.counted_ids:
                    current_count += 1
                    self.counted_ids.add(track_id)
                    logging.info(f"新增计数，ID: {track_id}, 总计数: {self.total_count + current_count}")

        return frame, current_count

    def detect_objects(self, frame):
        if not self.model:
            return []

        try:
            results = self.model.track(
                source=frame,
                conf=self.params['model_conf'],
                iou=self.params['model_nms'],
                imgsz=self.params['model_size'],
                verbose=False,
                device=self.device,
                tracker="bytetrack.yaml",  # 使用ByteTrack跟踪器
                persist=True,              # 保持跟踪状态
                amp=True                    # 启用自动混合精度
            )

            boxes = []
            track_ids = []
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                boxes_array = result.boxes.xywh.cpu().numpy()
                if result.boxes.id is not None:
                    track_ids.extend(result.boxes.id.cpu().numpy().astype(int))
                else:
                    track_ids.extend([-1] * len(boxes_array))
                
                for box in boxes_array:
                    x_center, y_center, width, height = box
                    x = x_center - width/2
                    y = y_center - height/2
                    boxes.append([x, y, width, height])

            return list(zip(boxes, track_ids))
        except Exception as e:
            error_msg = f"检测出错: {str(e)}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)
            return []

    def stop(self):
        with QMutexLocker(self.mutex):
            logging.info("正在停止视频线程...")
            self.running = False
            self.pause = False
            if self.cap and self.cap.isOpened():
                logging.info("释放视频资源...")
                self.cap.release()
            if self.model:
                logging.info("释放模型资源...")
                del self.model
                self.model = None
                torch.cuda.empty_cache()
            logging.info("视频线程停止完成")
            self.quit()
    
    def update_params(self, param_name, value):
        """更新检测参数"""
        with QMutexLocker(self.mutex):
            if param_name in self.params:
                self.params[param_name] = value
                self.params_updated.emit(self.params)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = '目标检测计数系统'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 720
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap.connect(self.update_image)
        self.video_thread.update_count.connect(self.update_count)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.model_loaded.connect(self.on_model_loaded)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧视频显示区域
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        layout.addWidget(self.image_label)

        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel)

        # 计数显示
        count_group = QGroupBox("计数信息")
        count_layout = QVBoxLayout()
        self.total_count_label = QLabel("总计数: 0")
        self.current_count_label = QLabel("当前计数: 0")
        self.fps_label = QLabel("处理时间: 0 ms")
        count_layout.addWidget(self.total_count_label)
        count_layout.addWidget(self.current_count_label)
        count_layout.addWidget(self.fps_label)
        count_group.setLayout(count_layout)
        control_layout.addWidget(count_group)

        # 模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择模型文件...")
        self.model_path_button = QPushButton("浏览")
        self.model_path_button.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_path_button)
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)

        # 视频源选择
        source_group = QGroupBox("视频源设置")
        source_layout = QVBoxLayout()
        
        # 视频源类型选择
        source_type_layout = QHBoxLayout()
        self.camera_radio = QRadioButton("摄像头")
        self.video_radio = QRadioButton("视频文件")
        self.camera_radio.setChecked(True)
        source_type_layout.addWidget(self.camera_radio)
        source_type_layout.addWidget(self.video_radio)
        source_layout.addLayout(source_type_layout)
        
        # 摄像头选择
        self.camera_group = QWidget()
        camera_layout = QVBoxLayout(self.camera_group)
        self.camera_combo = QComboBox()
        self.camera_combo.setPlaceholderText("选择摄像头...")
        self.refresh_camera_button = QPushButton("刷新摄像头列表")
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_camera_button)
        source_layout.addWidget(self.camera_group)
        
        # 视频文件选择
        self.video_group = QWidget()
        video_layout = QVBoxLayout(self.video_group)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        self.video_path_button = QPushButton("浏览")
        self.video_path_button.clicked.connect(self.select_video)
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(self.video_path_button)
        source_layout.addWidget(self.video_group)
        
        # 连接信号
        self.camera_radio.toggled.connect(self.on_source_type_changed)
        self.video_radio.toggled.connect(self.on_source_type_changed)
        
        source_group.setLayout(source_layout)
        control_layout.addWidget(source_group)
        
        # 初始化视频源界面
        self.refresh_cameras()
        self.on_source_type_changed()

        # 参数调整
        params_group = QGroupBox("参数调整")
        params_layout = QVBoxLayout()
        
        # 检测置信度
        conf_layout = QHBoxLayout()
        conf_label = QLabel("检测置信度:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.video_thread.params['model_conf'] * 100))
        self.conf_value = QLabel(f"{self.video_thread.params['model_conf']:.2f}")
        self.conf_slider.valueChanged.connect(lambda v: self.update_param('model_conf', v/100))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"{v/100:.2f}"))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        params_layout.addLayout(conf_layout)
        
        # NMS阈值
        nms_layout = QHBoxLayout()
        nms_label = QLabel("NMS阈值:")
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(1, 99)
        self.nms_slider.setValue(int(self.video_thread.params['model_nms'] * 100))
        self.nms_value = QLabel(f"{self.video_thread.params['model_nms']:.2f}")
        self.nms_slider.valueChanged.connect(lambda v: self.update_param('model_nms', v/100))
        self.nms_slider.valueChanged.connect(lambda v: self.nms_value.setText(f"{v/100:.2f}"))
        nms_layout.addWidget(nms_label)
        nms_layout.addWidget(self.nms_slider)
        nms_layout.addWidget(self.nms_value)
        params_layout.addLayout(nms_layout)
        
        # 跟踪距离
        track_dist_layout = QHBoxLayout()
        track_dist_label = QLabel("跟踪距离:")
        self.track_dist_slider = QSlider(Qt.Horizontal)
        self.track_dist_slider.setRange(10, 200)
        self.track_dist_slider.setValue(self.video_thread.params['track_dist'])
        self.track_dist_value = QLabel(f"{self.video_thread.params['track_dist']}")
        self.track_dist_slider.valueChanged.connect(lambda v: self.update_param('track_dist', v))
        self.track_dist_slider.valueChanged.connect(lambda v: self.track_dist_value.setText(str(v)))
        track_dist_layout.addWidget(track_dist_label)
        track_dist_layout.addWidget(self.track_dist_slider)
        track_dist_layout.addWidget(self.track_dist_value)
        params_layout.addLayout(track_dist_layout)
        
        # 检测底部区域
        bottom_layout = QHBoxLayout()
        bottom_label = QLabel("底部区域:")
        self.bottom_slider = QSlider(Qt.Horizontal)
        self.bottom_slider.setRange(5, 100)
        self.bottom_slider.setValue(self.video_thread.params['detect_bottom'])
        self.bottom_value = QLabel(f"{self.video_thread.params['detect_bottom']}")
        self.bottom_slider.valueChanged.connect(lambda v: self.update_param('detect_bottom', v))
        self.bottom_slider.valueChanged.connect(lambda v: self.bottom_value.setText(str(v)))
        bottom_layout.addWidget(bottom_label)
        bottom_layout.addWidget(self.bottom_slider)
        bottom_layout.addWidget(self.bottom_value)
        params_layout.addLayout(bottom_layout)
        
        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)
        
        # 控制按钮
        button_group = QGroupBox("控制")
        button_layout = QVBoxLayout()
        self.start_button = QPushButton("开始")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("结束")
        self.reset_button = QPushButton("重置")
        self.start_button.clicked.connect(self.start_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.reset_button.clicked.connect(self.reset_count)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        button_group.setLayout(button_layout)
        control_layout.addWidget(button_group)

        # 移除init_video_thread的调用，因为已经在__init__中完成初始化
        # self.init_video_thread()

    def init_video_thread(self):
        # 移除此方法，因为不再需要
        pass

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pt *.pth)")
        if file_name:
            self.model_path_edit.setText(file_name)
            QApplication.processEvents()  # 处理待处理的事件
            self.video_thread.load_model(file_name)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path_edit.setText(file_name)
            self.video_radio.setChecked(True)

    def refresh_cameras(self):
        """刷新摄像头列表"""
        self.camera_combo.clear()
        for idx, info in self.video_thread.available_cameras.items():
            self.camera_combo.addItem(
                f"摄像头 {idx} ({info['resolution']}, {info['fps']}fps)",
                idx
            )
    
    def on_source_type_changed(self):
        """处理视频源类型切换"""
        is_camera = self.camera_radio.isChecked()
        self.camera_group.setVisible(is_camera)
        self.video_group.setVisible(not is_camera)
    
    def start_video(self):
        if not self.video_thread.model:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return

        try:
            if self.camera_radio.isChecked():
                if self.camera_combo.currentData() is None:
                    raise ValueError("请选择摄像头")
                source = self.camera_combo.currentData()
            else:
                source = self.video_path_edit.text()
                if not source:
                    raise ValueError("请选择视频文件")

            QApplication.processEvents()  # 处理待处理的事件
            self.video_thread.set_source(source)
            self.video_thread.running = True
            self.video_thread.pause = False
            self.video_thread.condition.wakeAll()
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            
            # 禁用源选择控件
            self.camera_radio.setEnabled(False)
            self.video_radio.setEnabled(False)
            self.camera_combo.setEnabled(False)
            self.video_path_edit.setEnabled(False)
            self.video_path_button.setEnabled(False)
            self.refresh_camera_button.setEnabled(False)

        except Exception as e:
            self.show_error(str(e))

    def pause_video(self):
        if self.video_thread.running:
            self.video_thread.pause = not self.video_thread.pause
            self.pause_button.setText("继续" if self.video_thread.pause else "暂停")

    def reset_count(self):
        self.video_thread.reset_counters()
        self.total_count_label.setText("总计数: 0")
        self.current_count_label.setText("当前计数: 0")

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    @pyqtSlot(int, int, float)
    def update_count(self, total_count, current_count, elapsed):
        self.total_count_label.setText(f"总计数: {total_count}")
        self.current_count_label.setText(f"当前计数: {current_count}")
        self.fps_label.setText(f"处理时间: {elapsed*1000:.1f} ms")

    @pyqtSlot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    @pyqtSlot(bool)
    def on_model_loaded(self, success):
        if success:
            QMessageBox.information(self, "提示", "模型加载成功")
        self.start_button.setEnabled(success)

    def update_param(self, param_name, value):
        """更新参数值"""
        self.video_thread.update_params(param_name, value)
    
    def stop_video(self):
        """结束当前检测"""
        if self.video_thread:
            self.video_thread.stop()
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("暂停")
            
            # 重新启用源选择控件
            self.camera_radio.setEnabled(True)
            self.video_radio.setEnabled(True)
            self.camera_combo.setEnabled(True)
            self.video_path_edit.setEnabled(True)
            self.video_path_button.setEnabled(True)
            self.refresh_camera_button.setEnabled(True)
            
            # 清空显示
            self.image_label.clear()
            self.total_count_label.setText("总计数: 0")
            self.current_count_label.setText("当前计数: 0")
            self.fps_label.setText("处理时间: 0 ms")

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())