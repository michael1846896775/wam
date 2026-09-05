import sys
import os
import configparser
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QMessageBox, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont

class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频处理工具链（抽帧+位姿+去背景+3D重建）")
        self.setGeometry(300, 200, 900, 700)
        
        # Windows桌面路径自动识别
        self.windows_desktop_path = self.get_windows_desktop_path()
        
        # 存储路径与配置
        self.folder_path = ""
        self.config = configparser.ConfigParser()
        self.config_file = "video_processor_config.ini"
        self.load_saved_path()

        # 进程管理
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.update_output)
        self.process.readyReadStandardError.connect(self.update_error)
        self.process.finished.connect(self.process_finished)
        
        # 当前执行阶段标记
        self.current_stage = "frame_extract"  # frame_extract / pose_extract / bg_remove / gs_train / transfer_ply
        
        self.init_ui()
    
    def get_windows_desktop_path(self):
        try:
            windows_user = os.environ.get("USERPROFILE", "").split("\\")[-1]
            if not windows_user:
                return "/mnt"
            default_desktop = f"/mnt/c/Users/{windows_user}/Desktop"
            return default_desktop if os.path.exists(default_desktop) else "/mnt"
        except Exception as e:
            print(f"获取桌面路径失败：{e}")
            return "/mnt"
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("视频处理工具链（抽帧+位姿提取+背景去除+3D重建+点云复制）")
        title_font = QFont("Noto Sans CJK SC", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 路径显示
        self.path_label = QLabel(
            f"✅ 已选择文件夹：{self.folder_path}" if self.folder_path 
            else f"❌❌ 未选择文件夹（默认打开桌面路径）"
        )
        path_font = QFont("Noto Sans CJK SC", 10)
        self.path_label.setFont(path_font)
        self.path_label.setWordWrap(True)
        main_layout.addWidget(self.path_label)
        
        # 功能选择复选框
        self.pose_checkbox = QCheckBox("抽帧完成后自动提取图片位姿")
        self.pose_checkbox.setFont(QFont("Noto Sans CJK SC", 10))
        self.pose_checkbox.setChecked(True)
        
        self.bg_remove_checkbox = QCheckBox("位姿提取完成后自动去除图片背景")
        self.bg_remove_checkbox.setFont(QFont("Noto Sans CJK SC", 10))
        self.bg_remove_checkbox.setChecked(True)
        
        self.gs_train_checkbox = QCheckBox("背景去除完成后自动进行3D高斯重建")
        self.gs_train_checkbox.setFont(QFont("Noto Sans CJK SC", 10))
        self.gs_train_checkbox.setChecked(True)
        
        # 新增点云复制复选框
        self.transfer_ply_checkbox = QCheckBox("3D高斯重建完成后自动复制点云文件")
        self.transfer_ply_checkbox.setFont(QFont("Noto Sans CJK SC", 10))
        self.transfer_ply_checkbox.setChecked(True)
        
        main_layout.addWidget(self.pose_checkbox)
        main_layout.addWidget(self.bg_remove_checkbox)
        main_layout.addWidget(self.gs_train_checkbox)
        main_layout.addWidget(self.transfer_ply_checkbox)  # 添加到界面
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(40)
        
        # 选择文件夹按钮
        self.select_btn = QPushButton("选择文件夹（默认打开桌面）")
        select_font = QFont("Noto Sans CJK SC", 10)
        self.select_btn.setFont(select_font)
        self.select_btn.setMinimumSize(180, 40)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.select_btn.clicked.connect(self.select_folder)
        button_layout.addWidget(self.select_btn)
        
        # 开始处理按钮
        self.run_btn = QPushButton("开始完整处理流程")
        self.run_btn.setFont(select_font)
        self.run_btn.setMinimumSize(150, 40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.run_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.run_btn)
        
        main_layout.addLayout(button_layout)
        main_layout.setAlignment(button_layout, Qt.AlignCenter)
        
        # 操作说明
        info_label = QLabel("""
📌📌 操作说明：
1. 选择包含视频的文件夹（默认打开Windows桌面）
2. 可选择是否自动执行"位姿提取"、"背景去除"、"3D重建"和"点云复制"步骤
3. 点击"开始完整处理流程"启动以下操作：
   - 第一步：视频抽帧（chouzhen.py）
   - 第二步（可选）：提取图片位姿（vggsfm.sh）
   - 第三步（可选）：去除图片背景（sam2.sh）
   - 第四步（可选）：3D高斯重建（gaussian_splatting.sh）
   - 第五步（可选）：复制点云文件（transfer_ply.py）
4. 所有步骤的运行日志将实时显示在下方区域
        """)
        info_font = QFont("Noto Sans CJK SC", 9)
        info_label.setFont(info_font)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #7f8c8d;")
        main_layout.addWidget(info_label)
        
        # 终端输出显示区域
        self.output_text = QTextEdit()
        self.output_text.setFont(QFont("Courier New", 10))
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ddd;")
        main_layout.addWidget(self.output_text)
    
    def load_saved_path(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.config.read_file(f)
                if "Paths" in self.config and "last_folder" in self.config["Paths"]:
                    self.folder_path = self.config["Paths"]["last_folder"]
            except Exception as e:
                QMessageBox.warning(self, "配置文件警告", f"读取历史路径失败：{str(e)}")
    
    def save_path(self):
        try:
            if not self.config.has_section("Paths"):
                self.config.add_section("Paths")
            self.config.set("Paths", "last_folder", self.folder_path)
            with open(self.config_file, "w", encoding="utf-8") as f:
                self.config.write(f)
        except Exception as e:
            QMessageBox.critical(self, "配置文件错误", f"保存路径失败：{str(e)}")
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "选择文件夹（默认打开桌面）", 
            self.windows_desktop_path
        )
        if folder:
            self.folder_path = folder
            win_style_path = folder.replace("/mnt/c", "C:").replace("/", "\\")
            display_path = win_style_path if len(win_style_path) <= 50 else f"{win_style_path[:50]}..."
            self.path_label.setText(f"✅ 已选择文件夹：{display_path}")
            self.save_path()
    
    def start_processing(self):
        if not self.folder_path or not os.path.isdir(self.folder_path):
            QMessageBox.critical(self, "错误", "❌❌ 请先选择有效的文件夹！")
            return
        
        self.output_text.clear()
        self.run_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.pose_checkbox.setEnabled(False)
        self.bg_remove_checkbox.setEnabled(False)
        self.gs_train_checkbox.setEnabled(False)
        self.transfer_ply_checkbox.setEnabled(False)  # 禁用新复选框
        
        # 开始第一步：抽帧
        self.current_stage = "frame_extract"
        self.output_text.insertPlainText("===== 【第一步】开始视频抽帧 =====")
        self.run_frame_extraction()
    
    # 第一步：视频抽帧
    def run_frame_extraction(self):
        command = [
            "python", 
            "chouzhen.py", 
            "--input", self.folder_path, 
            "--output", self.folder_path
        ]
        self.process.setWorkingDirectory("/home/wangam")
        self.process.start("bash", ["-c", " ".join(command)])
    
    # 第二步：提取图片位姿
    def run_pose_extraction(self):
        self.current_stage = "pose_extract"
        self.output_text.insertPlainText("\n\n===== 【第二步】开始提取图片位姿 =====")
        
        # 使用独立的shell脚本
        script_path = "/home/wangam/vggsfm.sh"
        command = f"bash {script_path} \"{self.folder_path}\""
        self.process.start("bash", ["-c", command])
    
    # 第三步：去除图片背景
    def run_bg_removal(self):
        self.current_stage = "bg_remove"
        self.output_text.insertPlainText("\n\n===== 【第三步】开始去除图片背景 =====")
        
        # 使用独立的shell脚本
        script_path = "/home/wangam/sam2.sh"
        command = f"bash {script_path} \"{self.folder_path}\""
        self.process.start("bash", ["-c", command])
    
    # 第四步：3D高斯重建
    def run_gs_training(self):
        self.current_stage = "gs_train"
        self.output_text.insertPlainText("\n\n===== 【第四步】开始3D高斯重建 =====")
        
        # 使用独立的shell脚本
        script_path = "/home/wangam/gaussian_splatting.sh"
        command = f"bash {script_path} \"{self.folder_path}\""
        self.process.start("bash", ["-c", command])
    
    # 新增第五步：复制点云文件
    def run_transfer_ply(self):
        self.current_stage = "transfer_ply"
        self.output_text.insertPlainText("\n\n===== 【第五步】开始复制点云文件 =====")
        
        # 执行点云复制命令
        command = [
            "python", 
            "transfer_ply.py", 
            "--source_dir", self.folder_path
        ]
        self.process.setWorkingDirectory("/home/wangam")
        self.process.start("bash", ["-c", " ".join(command)])
    
    def update_output(self):
        data = self.process.readAllStandardOutput()
        output = str(data, encoding="utf-8")
        self.output_text.insertPlainText(output)
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())
    
    def update_error(self):
        data = self.process.readAllStandardError()
        error = str(data, encoding="utf-8")
        self.output_text.insertPlainText(error)
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())
    
    def process_finished(self, exit_code):
        # 处理流程衔接逻辑
        if self.current_stage == "frame_extract":
            # 抽帧完成后处理
            if exit_code == 0:
                self.output_text.insertPlainText("\n===== 【第一步】视频抽帧完成 =====")
                # 判断是否需要执行位姿提取
                if self.pose_checkbox.isChecked():
                    self.run_pose_extraction()
                else:
                    # 不执行位姿提取，判断是否直接执行背景去除
                    if self.bg_remove_checkbox.isChecked():
                        self.run_bg_removal()
                    else:
                        # 不执行背景去除，判断是否直接执行3D重建
                        if self.gs_train_checkbox.isChecked():
                            self.run_gs_training()
                        else:
                            # 检查是否需要执行点云复制（即使前面步骤都没执行）
                            if self.transfer_ply_checkbox.isChecked():
                                self.run_transfer_ply()
                            else:
                                self.restore_ui_state()
            else:
                self.output_text.insertPlainText(f"\n===== 【第一步】视频抽帧失败，退出码：{exit_code} =====")
                self.restore_ui_state()
        
        elif self.current_stage == "pose_extract":
            # 位姿提取完成后处理
            if exit_code == 0:
                self.output_text.insertPlainText("\n===== 【第二步】图片位姿提取完成 =====")
                # 判断是否需要执行背景去除
                if self.bg_remove_checkbox.isChecked():
                    self.run_bg_removal()
                else:
                    # 不执行背景去除，判断是否直接执行3D重建
                    if self.gs_train_checkbox.isChecked():
                        self.run_gs_training()
                    else:
                        # 检查是否需要执行点云复制
                        if self.transfer_ply_checkbox.isChecked():
                            self.run_transfer_ply()
                        else:
                            self.restore_ui_state()
            else:
                self.output_text.insertPlainText(f"\n===== 【第二步】图片位姿提取失败，退出码：{exit_code} =====")
                self.restore_ui_state()
        
        elif self.current_stage == "bg_remove":
            # 背景去除完成后处理
            if exit_code == 0:
                self.output_text.insertPlainText("\n===== 【第三步】图片背景去除完成 =====")
                # 判断是否需要执行3D重建
                if self.gs_train_checkbox.isChecked():
                    self.run_gs_training()
                else:
                    # 检查是否需要执行点云复制
                    if self.transfer_ply_checkbox.isChecked():
                        self.run_transfer_ply()
                    else:
                        self.restore_ui_state()
            else:
                self.output_text.insertPlainText(f"\n===== 【第三步】图片背景去除失败，退出码：{exit_code} =====")
                self.restore_ui_state()
        
        elif self.current_stage == "gs_train":
            # 3D重建完成后处理
            if exit_code == 0:
                self.output_text.insertPlainText("\n===== 【第四步】3D高斯重建完成 =====")
                # 判断是否需要执行点云复制
                if self.transfer_ply_checkbox.isChecked():
                    self.run_transfer_ply()
                else:
                    self.restore_ui_state()
            else:
                self.output_text.insertPlainText(f"\n===== 【第四步】3D高斯重建失败，退出码：{exit_code} =====")
                self.restore_ui_state()
        
        # 新增点云复制完成后的处理逻辑
        elif self.current_stage == "transfer_ply":
            if exit_code == 0:
                self.output_text.insertPlainText("\n===== 【第五步】点云文件复制完成 =====")
            else:
                self.output_text.insertPlainText(f"\n===== 【第五步】点云文件复制失败，退出码：{exit_code} =====")
            self.restore_ui_state()
    
    def restore_ui_state(self):
        # 恢复界面交互状态
        self.run_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.pose_checkbox.setEnabled(True)
        self.bg_remove_checkbox.setEnabled(True)
        self.gs_train_checkbox.setEnabled(True)
        self.transfer_ply_checkbox.setEnabled(True)  # 恢复新复选框状态
        self.output_text.insertPlainText("\n\n所有选中的处理步骤已完成！")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Noto Sans CJK SC")
    app.setFont(font)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())