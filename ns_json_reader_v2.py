import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QLineEdit, QTextEdit, QMessageBox, 
                            QFileDialog, QGroupBox, QSplitter, QFrame, QStatusBar,
                            QMenu, QMenuBar, QDialog, QRadioButton, QButtonGroup,
                            QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont, QTextCursor, QTextCharFormat, QColor, QTextDocument


class SearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("搜尋")
        self.setMinimumWidth(400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 搜尋詞輸入區域
        search_layout = QHBoxLayout()
        search_label = QLabel("搜尋詞:")
        self.search_input = QLineEdit()
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # 搜尋範圍選擇區域
        self.radio_prompt = QRadioButton("僅搜尋 Prompt")
        self.radio_completion = QRadioButton("僅搜尋 Completion")
        self.radio_both = QRadioButton("兩者都搜尋")
        self.radio_both.setChecked(True)
        
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.radio_prompt)
        radio_layout.addWidget(self.radio_completion)
        radio_layout.addWidget(self.radio_both)
        layout.addLayout(radio_layout)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        self.search_button = QPushButton("搜尋")
        self.search_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.search_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
    def get_search_info(self):
        term = self.search_input.text().strip()
        
        if self.radio_prompt.isChecked():
            scope = "prompt"
        elif self.radio_completion.isChecked():
            scope = "completion"
        else:
            scope = "both"
            
        return term, scope


class SearchResultsDialog(QDialog):
    def __init__(self, term, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"搜尋結果: '{term}'")
        self.setMinimumSize(500, 400)
        self.results = results
        self.selected_index = None
        self.setup_ui(term)
        
    def setup_ui(self, term):
        layout = QVBoxLayout(self)
        
        # 結果標籤
        result_label = QLabel(f"找到 {len(self.results)} 個匹配項:")
        layout.addWidget(result_label)
        
        # 結果列表
        self.result_list = QListWidget()
        for idx, preview in self.results:
            self.result_list.addItem(f"{idx+1}: {preview}")
        layout.addWidget(self.result_list)
        
        # 按鈕
        button_layout = QHBoxLayout()
        go_button = QPushButton("前往所選項目")
        go_button.clicked.connect(self.go_to_selected)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(go_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # 雙擊也能選擇
        self.result_list.itemDoubleClicked.connect(self.go_to_selected)
        
    def go_to_selected(self):
        current_row = self.result_list.currentRow()
        if current_row >= 0:
            self.selected_index = self.results[current_row][0]
            self.accept()


class NSJSONReader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NS-LLM JSON Reader")
        self.resize(1200, 800)
        
        self.current_file = None
        self.data = []
        self.current_index = 0
        self.modified = False
        
        self.setup_ui()
        self.create_menu()
        self.create_shortcuts()
        
        self.update_ui_state()
        
    def setup_ui(self):
        # 中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 導航區域
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ 上一個")
        self.prev_button.clicked.connect(self.prev_item)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("下一個 ▶")
        self.next_button.clicked.connect(self.next_item)
        nav_layout.addWidget(self.next_button)
        
        self.index_label = QLabel("0/0")
        nav_layout.addWidget(self.index_label)
        
        nav_layout.addWidget(QLabel("跳轉至:"))
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(50)
        self.jump_input.returnPressed.connect(self.jump_to_index)
        nav_layout.addWidget(self.jump_input)
        
        jump_button = QPushButton("前往")
        jump_button.clicked.connect(self.jump_to_index)
        nav_layout.addWidget(jump_button)
        
        self.modified_label = QLabel("")
        nav_layout.addWidget(self.modified_label)
        
        nav_layout.addStretch()
        
        add_button = QPushButton("新增項目")
        add_button.clicked.connect(self.add_new_item)
        nav_layout.addWidget(add_button)
        
        delete_button = QPushButton("刪除此項")
        delete_button.clicked.connect(self.delete_item)
        nav_layout.addWidget(delete_button)
        
        main_layout.addLayout(nav_layout)
        
        # 內容區域
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.setChildrenCollapsible(False)
        
        # Prompt 區域
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_text = QTextEdit()
        self.prompt_text.setFont(QFont("微軟雅黑", 10))
        self.prompt_text.textChanged.connect(lambda: self.set_modified(True))
        prompt_layout.addWidget(self.prompt_text)
        
        # Completion 區域
        completion_group = QGroupBox("Completion")
        completion_layout = QVBoxLayout(completion_group)
        self.completion_text = QTextEdit()
        self.completion_text.setFont(QFont("微軟雅黑", 10))
        self.completion_text.textChanged.connect(lambda: self.set_modified(True))
        completion_layout.addWidget(self.completion_text)
        
        content_splitter.addWidget(prompt_group)
        content_splitter.addWidget(completion_group)
        content_splitter.setSizes([200, 400])
        
        main_layout.addWidget(content_splitter)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.apply_button = QPushButton("應用更改 (Ctrl+Enter)")
        self.apply_button.clicked.connect(self.apply_changes)
        button_layout.addWidget(self.apply_button)
        
        main_layout.addLayout(button_layout)
        
        # 狀態欄
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就緒")
        
    def create_menu(self):
        menubar = self.menuBar()
        
        # 檔案選單
        file_menu = menubar.addMenu("檔案")
        
        open_action = QAction("開啟舊檔 (Ctrl+O)", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("儲存 (Ctrl+S)", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("另存新檔 (Ctrl+Shift+S)", self)
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 編輯選單
        edit_menu = menubar.addMenu("編輯")
        
        find_action = QAction("搜尋 (Ctrl+F)", self)
        find_action.triggered.connect(self.show_search_dialog)
        edit_menu.addAction(find_action)
        
        edit_menu.addSeparator()
        
        add_action = QAction("新增項目", self)
        add_action.triggered.connect(self.add_new_item)
        edit_menu.addAction(add_action)
        
        delete_action = QAction("刪除此項", self)
        delete_action.triggered.connect(self.delete_item)
        edit_menu.addAction(delete_action)
    
    def create_shortcuts(self):
        # 檔案操作快捷鍵
        open_shortcut = QAction(self)
        open_shortcut.setShortcut("Ctrl+O")
        open_shortcut.triggered.connect(self.open_file)
        self.addAction(open_shortcut)
        
        save_shortcut = QAction(self)
        save_shortcut.setShortcut("Ctrl+S")
        save_shortcut.triggered.connect(self.save_file)
        self.addAction(save_shortcut)
        
        save_as_shortcut = QAction(self)
        save_as_shortcut.setShortcut("Ctrl+Shift+S")
        save_as_shortcut.triggered.connect(self.save_file_as)
        self.addAction(save_as_shortcut)
        
        # 編輯操作快捷鍵
        find_shortcut = QAction(self)
        find_shortcut.setShortcut("Ctrl+F")
        find_shortcut.triggered.connect(self.show_search_dialog)
        self.addAction(find_shortcut)
        
        apply_shortcut = QAction(self)
        apply_shortcut.setShortcut("Ctrl+Return")
        apply_shortcut.triggered.connect(self.apply_changes)
        self.addAction(apply_shortcut)
        
        # 導航快捷鍵
        prev_shortcut = QAction(self)
        prev_shortcut.setShortcut("Ctrl+Left")
        prev_shortcut.triggered.connect(self.prev_item)
        self.addAction(prev_shortcut)
        
        next_shortcut = QAction(self)
        next_shortcut.setShortcut("Ctrl+Right")
        next_shortcut.triggered.connect(self.next_item)
        self.addAction(next_shortcut)
    
    def set_modified(self, value):
        self.modified = value
        if value:
            self.modified_label.setText("* 已修改")
        else:
            self.modified_label.setText("")
    
    def update_ui_state(self):
        has_data = len(self.data) > 0
        
        self.prev_button.setEnabled(has_data and self.current_index > 0)
        self.next_button.setEnabled(has_data and self.current_index < len(self.data) - 1)
        self.prompt_text.setEnabled(has_data)
        self.completion_text.setEnabled(has_data)
        self.apply_button.setEnabled(has_data)
        self.jump_input.setEnabled(has_data)
        
        if has_data:
            self.index_label.setText(f"{self.current_index + 1}/{len(self.data)}")
        else:
            self.index_label.setText("0/0")
    
    def open_file(self):
        if self.modified:
            reply = QMessageBox.question(self, "儲存更改", "是否儲存當前更改?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                if not self.save_file():
                    return  # 如果保存失敗，取消打开操作
            elif reply == QMessageBox.StandardButton.Cancel:
                return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "開啟檔案", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                
                self.current_file = file_path
                self.current_index = 0
                self.set_modified(False)
                
                self.update_display()
                self.update_ui_state()
                
                self.setWindowTitle(f"NS-LLM JSON Reader - {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"已開啟: {file_path} | {len(self.data)} 個項目")
                
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法開啟檔案: {str(e)}")
    
    def save_file(self):
        if not self.current_file:
            return self.save_file_as()
        
        try:
            # 先應用當前更改
            if self.modified:
                self.apply_changes_without_message()
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            self.set_modified(False)
            self.statusBar.showMessage(f"已儲存: {self.current_file}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法儲存檔案: {str(e)}")
            return False
    
    def save_file_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "另存新檔", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.current_file = file_path
            if self.save_file():
                self.setWindowTitle(f"NS-LLM JSON Reader - {os.path.basename(file_path)}")
                return True
        
        return False
    
    def update_display(self):
        if not self.data:
            self.prompt_text.clear()
            self.completion_text.clear()
            return
        
        # 確保 current_index 在有效範圍內
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.data):
            self.current_index = len(self.data) - 1
        
        current_item = self.data[self.current_index]
        
        # 阻止 textChanged 信號觸發 set_modified
        self.prompt_text.blockSignals(True)
        self.completion_text.blockSignals(True)
        
        # 更新文字區域
        self.prompt_text.setPlainText(current_item.get("prompt", ""))
        self.completion_text.setPlainText(current_item.get("completion", ""))
        
        self.prompt_text.blockSignals(False)
        self.completion_text.blockSignals(False)
        
        # 更新索引標籤
        self.index_label.setText(f"{self.current_index + 1}/{len(self.data)}")
        
        # 重置修改狀態
        self.set_modified(False)
    
    def apply_changes_without_message(self):
        if not self.data:
            return
        
        # 獲取當前編輯的數據
        prompt = self.prompt_text.toPlainText().strip()
        completion = self.completion_text.toPlainText().strip()
        
        # 更新數據
        self.data[self.current_index]["prompt"] = prompt
        self.data[self.current_index]["completion"] = completion
        
        self.set_modified(False)
    
    def apply_changes(self):
        if not self.data:
            return
            
        self.apply_changes_without_message()
        QMessageBox.information(self, "成功", "已套用更改!")
    
    def prev_item(self):
        if self.current_index > 0:
            # 如果有修改，先保存
            if self.modified:
                reply = QMessageBox.question(self, "儲存更改", "是否儲存對當前項目的更改?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Yes:
                    self.apply_changes_without_message()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return
                else:
                    self.set_modified(False)
            
            self.current_index -= 1
            self.update_display()
            self.update_ui_state()
    
    def next_item(self):
        if self.current_index < len(self.data) - 1:
            # 如果有修改，先保存
            if self.modified:
                reply = QMessageBox.question(self, "儲存更改", "是否儲存對當前項目的更改?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Yes:
                    self.apply_changes_without_message()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return
                else:
                    self.set_modified(False)
            
            self.current_index += 1
            self.update_display()
            self.update_ui_state()
    
    def jump_to_index(self):
        try:
            index = int(self.jump_input.text()) - 1  # 用户輸入從1開始，但索引從0開始
            
            if 0 <= index < len(self.data):
                # 如果有修改，先保存
                if self.modified:
                    reply = QMessageBox.question(self, "儲存更改", "是否儲存對當前項目的更改?",
                                               QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                    if reply == QMessageBox.StandardButton.Yes:
                        self.apply_changes_without_message()
                    elif reply == QMessageBox.StandardButton.Cancel:
                        return
                    else:
                        self.set_modified(False)
                
                self.current_index = index
                self.update_display()
                self.update_ui_state()
            else:
                QMessageBox.warning(self, "錯誤", f"索引必須在 1 和 {len(self.data)} 之間")
                
        except ValueError:
            QMessageBox.warning(self, "錯誤", "請輸入有效的數字")
    
    def add_new_item(self):
        # 如果有修改，先保存
        if self.modified:
            reply = QMessageBox.question(self, "儲存更改", "是否儲存對當前項目的更改再新增?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.apply_changes_without_message()
            elif reply == QMessageBox.StandardButton.Cancel:
                return
            else:
                self.set_modified(False)
        
        # 創建新項
        new_item = {"prompt": "", "completion": ""}
        
        # 添加到數據中
        self.data.append(new_item)
        
        # 跳轉到新項
        self.current_index = len(self.data) - 1
        self.update_display()
        self.update_ui_state()
        self.set_modified(True)
    
    def delete_item(self):
        if not self.data:
            return
            
        reply = QMessageBox.question(self, "確認", "確定要刪除此項目?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del self.data[self.current_index]
            self.set_modified(True)
            
            if not self.data:
                self.prompt_text.clear()
                self.completion_text.clear()
                self.index_label.setText("0/0")
            else:
                if self.current_index >= len(self.data):
                    self.current_index = len(self.data) - 1
                self.update_display()
                
            self.update_ui_state()
    
    def show_search_dialog(self):
        dialog = SearchDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            term, scope = dialog.get_search_info()
            if term:
                self.search_in_data(term, scope)
    
    def search_in_data(self, term, scope):
        found_indexes = []
        
        for i, item in enumerate(self.data):
            prompt = item.get("prompt", "").lower()
            completion = item.get("completion", "").lower()
            term_lower = term.lower()
            
            is_found = False
            
            if scope == "prompt" or scope == "both":
                if term_lower in prompt:
                    prompt_preview = item.get("prompt", "")[:50].replace("\n", " ")
                    if len(item.get("prompt", "")) > 50:
                        prompt_preview += "..."
                    found_indexes.append((i, prompt_preview))
                    is_found = True
            
            if not is_found and (scope == "completion" or scope == "both"):
                if term_lower in completion:
                    prompt_preview = item.get("prompt", "")[:50].replace("\n", " ")
                    if len(item.get("prompt", "")) > 50:
                        prompt_preview += "..."
                    found_indexes.append((i, prompt_preview))
        
        if found_indexes:
            results_dialog = SearchResultsDialog(term, found_indexes, self)
            if results_dialog.exec() == QDialog.DialogCode.Accepted:
                if results_dialog.selected_index is not None:
                    # 如果有修改，先保存
                    if self.modified:
                        reply = QMessageBox.question(self, "儲存更改", "是否儲存對當前項目的更改?",
                                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                        if reply == QMessageBox.StandardButton.Yes:
                            self.apply_changes_without_message()
                        elif reply == QMessageBox.StandardButton.Cancel:
                            return
                        else:
                            self.set_modified(False)
                    
                    self.current_index = results_dialog.selected_index
                    self.update_display()
                    self.update_ui_state()
                    self.highlight_search_term(term)
        else:
            QMessageBox.information(self, "搜尋", f"未找到 '{term}' 的匹配項")
    
    def highlight_search_term(self, term):
        # 在 prompt 和 completion 中高亮搜尋詞
        self.highlight_in_text_edit(self.prompt_text, term)
        self.highlight_in_text_edit(self.completion_text, term)
    
    def highlight_in_text_edit(self, text_edit, term):
        cursor = text_edit.textCursor()
        cursor.setPosition(0)
        text_edit.setTextCursor(cursor)
        
        # 清除所有已有的格式
        cursor = QTextCursor(text_edit.document())
        cursor.select(QTextCursor.SelectionType.Document)
        format = QTextCharFormat()
        cursor.setCharFormat(format)
        
        # 高亮所有匹配的詞
        term_lower = term.lower()
        document = text_edit.document()
        find_cursor = QTextCursor(document)
        
        while True:
            find_cursor = document.find(term, find_cursor, QTextDocument.FindFlag.FindCaseSensitively)
            if find_cursor.isNull():
                break
                
            format = QTextCharFormat()
            format.setBackground(QColor("yellow"))
            find_cursor.mergeCharFormat(format)
    
    def closeEvent(self, event):
        if self.modified:
            reply = QMessageBox.question(self, "儲存更改", "是否儲存更改後退出?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                if self.save_file():
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NSJSONReader()
    window.show()
    sys.exit(app.exec())