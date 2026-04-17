import sys
import json
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QLineEdit, QTextEdit, QMessageBox, 
                            QFileDialog, QGroupBox, QSplitter, QFrame, QStatusBar,
                            QMenu, QMenuBar, QDialog, QRadioButton, QButtonGroup,
                            QListWidget, QListWidgetItem, QTabWidget, QCheckBox,
                            QComboBox, QSpinBox, QTextBrowser, QScrollArea, QToolBar)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, QSettings
from PyQt6.QtGui import (QAction, QFont, QTextCursor, QTextCharFormat, QColor, 
                        QTextDocument, QSyntaxHighlighter, QTextBlockFormat, QPalette)
import markdown
from markdown.extensions import codehilite, tables, toc


class ThemeManager:
    """主題管理器"""
    def __init__(self):
        self.is_dark_mode = False
        self.settings = QSettings('NSJSONReader', 'Theme')
        self.load_theme_preference()
    
    def load_theme_preference(self):
        """載入主題偏好設定"""
        self.is_dark_mode = self.settings.value('dark_mode', False, type=bool)
    
    def save_theme_preference(self):
        """保存主題偏好設定"""
        self.settings.setValue('dark_mode', self.is_dark_mode)
    
    def toggle_theme(self):
        """切換主題"""
        self.is_dark_mode = not self.is_dark_mode
        self.save_theme_preference()
    
    def get_editor_style(self):
        """獲取編輯器樣式"""
        if self.is_dark_mode:
            return """
                QTextEdit {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #555;
                    padding: 8px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 11pt;
                    selection-background-color: #3d4142;
                }
                QTextEdit:focus {
                    border: 2px solid #0078d4;
                }
            """
        else:
            return """
                QTextEdit {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #ccc;
                    padding: 8px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 11pt;
                    selection-background-color: #b3d7ff;
                }
                QTextEdit:focus {
                    border: 2px solid #0078d4;
                }
            """
    
    def get_browser_style(self):
        """獲取瀏覽器樣式"""
        if self.is_dark_mode:
            return """
                QTextBrowser {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #555;
                    padding: 10px;
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    font-size: 11pt;
                    selection-background-color: #3d4142;
                }
            """
        else:
            return """
                QTextBrowser {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #ccc;
                    padding: 10px;
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    font-size: 11pt;
                    selection-background-color: #b3d7ff;
                }
            """
    
    def get_markdown_css(self):
        """獲取Markdown CSS樣式"""
        if self.is_dark_mode:
            return """
                body { 
                    font-family: 'Microsoft YaHei', Arial, sans-serif; 
                    line-height: 1.6; 
                    color: #ffffff;
                    background-color: #2b2b2b;
                    margin: 0;
                    padding: 0;
                }
                h1, h2, h3, h4, h5, h6 { 
                    color: #66d9ef; 
                    margin-top: 20px; 
                    margin-bottom: 10px; 
                }
                h1 { 
                    border-bottom: 2px solid #555; 
                    padding-bottom: 5px; 
                }
                h2 { 
                    border-bottom: 1px solid #555; 
                    padding-bottom: 5px; 
                }
                code { 
                    background-color: #3e3e3e; 
                    color: #f92672;
                    padding: 2px 4px; 
                    border-radius: 3px; 
                    font-family: 'Consolas', monospace; 
                }
                pre { 
                    background-color: #1e1e1e; 
                    color: #ffffff;
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto;
                    border: 1px solid #555;
                }
                pre code {
                    background-color: transparent;
                    color: #ffffff;
                }
                blockquote { 
                    border-left: 4px solid #66d9ef; 
                    padding-left: 15px; 
                    margin-left: 0; 
                    color: #a6e22e;
                    background-color: #333;
                    padding: 10px;
                    border-radius: 3px;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    background-color: #333;
                }
                th, td { 
                    border: 1px solid #555; 
                    padding: 8px; 
                    text-align: left; 
                }
                th { 
                    background-color: #404040; 
                    color: #ffffff;
                }
                a { 
                    color: #66d9ef; 
                }
                a:hover {
                    color: #a6e22e;
                }
                ul, ol { 
                    padding-left: 20px; 
                }
                li {
                    margin: 5px 0;
                }
                strong {
                    color: #f92672;
                }
                em {
                    color: #a6e22e;
                }
                hr {
                    border: none;
                    border-top: 1px solid #555;
                    margin: 20px 0;
                }
                p {
                    margin: 8px 0;
                }
            """
        else:
            return """
                body { 
                    font-family: 'Microsoft YaHei', Arial, sans-serif; 
                    line-height: 1.6; 
                    color: #000000;
                    background-color: #ffffff;
                    margin: 0;
                    padding: 0;
                }
                h1, h2, h3, h4, h5, h6 { 
                    color: #2c3e50; 
                    margin-top: 20px; 
                    margin-bottom: 10px; 
                }
                h1 { 
                    border-bottom: 2px solid #ecf0f1; 
                    padding-bottom: 5px; 
                }
                h2 { 
                    border-bottom: 1px solid #ecf0f1; 
                    padding-bottom: 5px; 
                }
                code { 
                    background-color: #f8f9fa; 
                    color: #e74c3c;
                    padding: 2px 4px; 
                    border-radius: 3px; 
                    font-family: 'Consolas', monospace;
                    border: 1px solid #e9ecef;
                }
                pre { 
                    background-color: #f8f9fa; 
                    color: #2c3e50;
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto;
                    border: 1px solid #e9ecef;
                }
                pre code {
                    background-color: transparent;
                    color: #2c3e50;
                    border: none;
                }
                blockquote { 
                    border-left: 4px solid #3498db; 
                    padding-left: 15px; 
                    margin-left: 0; 
                    color: #7f8c8d;
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 3px;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    background-color: #ffffff;
                }
                th, td { 
                    border: 1px solid #dee2e6; 
                    padding: 8px; 
                    text-align: left; 
                }
                th { 
                    background-color: #f8f9fa; 
                    color: #495057;
                }
                a { 
                    color: #3498db; 
                }
                a:hover {
                    color: #2980b9;
                }
                ul, ol { 
                    padding-left: 20px; 
                }
                li {
                    margin: 5px 0;
                }
                strong {
                    color: #2c3e50;
                }
                em {
                    color: #7f8c8d;
                }
                hr {
                    border: none;
                    border-top: 1px solid #dee2e6;
                    margin: 20px 0;
                }
                p {
                    margin: 8px 0;
                }
            """
    
    def get_app_style(self):
        """獲取應用程式整體樣式"""
        if self.is_dark_mode:
            return """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QToolBar {
                    background-color: #2b2b2b;
                    border: none;
                    spacing: 3px;
                    padding: 2px;
                }
                QPushButton {
                    background-color: #404040;
                    border: 1px solid #555;
                    color: #ffffff;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 9pt;
                    min-height: 20px;
                    max-height: 26px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border: 1px solid #66d9ef;
                }
                QPushButton:pressed {
                    background-color: #333;
                }
                QPushButton:disabled {
                    background-color: #2a2a2a;
                    color: #666;
                    border: 1px solid #444;
                }
                QLineEdit {
                    background-color: #404040;
                    border: 1px solid #555;
                    color: #ffffff;
                    padding: 2px 4px;
                    border-radius: 3px;
                    max-height: 22px;
                    font-size: 9pt;
                }
                QLineEdit:focus {
                    border: 2px solid #66d9ef;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 9pt;
                }
                QTabWidget::pane {
                    border: 1px solid #555;
                    background-color: #2b2b2b;
                }
                QTabBar::tab {
                    background-color: #404040;
                    color: #ffffff;
                    border: 1px solid #555;
                    padding: 6px 12px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #66d9ef;
                    color: #000000;
                }
                QTabBar::tab:hover {
                    background-color: #4a4a4a;
                }
                QMenuBar {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border-bottom: 1px solid #555;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #404040;
                }
                QMenu {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #555;
                }
                QMenu::item {
                    padding: 6px 12px;
                }
                QMenu::item:selected {
                    background-color: #404040;
                }
                QStatusBar {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border-top: 1px solid #555;
                }
                QListWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #555;
                    alternate-background-color: #333;
                }
                QListWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #555;
                }
                QListWidget::item:selected {
                    background-color: #66d9ef;
                    color: #000000;
                }
                QCheckBox {
                    color: #ffffff;
                }
                QRadioButton {
                    color: #ffffff;
                }
                QDialog {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QSplitter::handle {
                    background-color: #555;
                    width: 4px;
                    height: 4px;
                }
                QSplitter::handle:hover {
                    background-color: #66d9ef;
                }
                QSplitter::handle:horizontal {
                    width: 4px;
                }
                QSplitter::handle:vertical {
                    height: 4px;
                }
            """
        else:
            return """
                QMainWindow {
                    background-color: #ffffff;
                    color: #000000;
                }
                QWidget {
                    background-color: #ffffff;
                    color: #000000;
                }
                QToolBar {
                    background-color: #f8f9fa;
                    border: none;
                    spacing: 3px;
                    padding: 2px;
                }
                QPushButton {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    color: #495057;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 9pt;
                    min-height: 20px;
                    max-height: 26px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border: 1px solid #3498db;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
                QPushButton:disabled {
                    background-color: #f8f9fa;
                    color: #6c757d;
                    border: 1px solid #dee2e6;
                }
                QLineEdit {
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    color: #495057;
                    padding: 2px 4px;
                    border-radius: 3px;
                    max-height: 22px;
                    font-size: 9pt;
                }
                QLineEdit:focus {
                    border: 2px solid #3498db;
                }
                QLabel {
                    color: #2c3e50;
                    font-size: 9pt;
                }
                QTabWidget::pane {
                    border: 1px solid #dee2e6;
                    background-color: #ffffff;
                }
                QTabBar::tab {
                    background-color: #f8f9fa;
                    color: #495057;
                    border: 1px solid #dee2e6;
                    padding: 6px 12px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #3498db;
                    color: #ffffff;
                }
                QTabBar::tab:hover {
                    background-color: #e9ecef;
                }
                QMenuBar {
                    background-color: #f8f9fa;
                    color: #495057;
                    border-bottom: 1px solid #dee2e6;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #e9ecef;
                }
                QMenu {
                    background-color: #ffffff;
                    color: #495057;
                    border: 1px solid #dee2e6;
                }
                QMenu::item {
                    padding: 6px 12px;
                }
                QMenu::item:selected {
                    background-color: #f8f9fa;
                }
                QStatusBar {
                    background-color: #f8f9fa;
                    color: #495057;
                    border-top: 1px solid #dee2e6;
                }
                QListWidget {
                    background-color: #ffffff;
                    color: #495057;
                    border: 1px solid #dee2e6;
                    alternate-background-color: #f8f9fa;
                }
                QListWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #f1f3f4;
                }
                QListWidget::item:selected {
                    background-color: #3498db;
                    color: #ffffff;
                }
                QCheckBox {
                    color: #495057;
                }
                QRadioButton {
                    color: #495057;
                }
                QDialog {
                    background-color: #ffffff;
                    color: #495057;
                }
                QSplitter::handle {
                    background-color: #dee2e6;
                    width: 4px;
                    height: 4px;
                }
                QSplitter::handle:hover {
                    background-color: #3498db;
                }
                QSplitter::handle:horizontal {
                    width: 4px;
                }
                QSplitter::handle:vertical {
                    height: 4px;
                }
            """


class MarkdownHighlighter(QSyntaxHighlighter):
    """Markdown語法高亮器"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.update_highlighting_rules()
    
    def update_highlighting_rules(self):
        """根據主題更新高亮規則"""
        self.highlighting_rules = []
        
        is_dark = self.theme_manager.is_dark_mode if self.theme_manager else False
        
        # 標題格式
        header_format = QTextCharFormat()
        header_format.setForeground(QColor("#66d9ef" if is_dark else "#2c3e50"))
        header_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((r'^#{1,6}\s.*', header_format))
        
        # 粗體
        bold_format = QTextCharFormat()
        bold_format.setFontWeight(QFont.Weight.Bold)
        bold_format.setForeground(QColor("#f92672" if is_dark else "#e74c3c"))
        self.highlighting_rules.append((r'\*\*.*?\*\*', bold_format))
        self.highlighting_rules.append((r'__.*?__', bold_format))
        
        # 斜體
        italic_format = QTextCharFormat()
        italic_format.setFontItalic(True)
        italic_format.setForeground(QColor("#a6e22e" if is_dark else "#27ae60"))
        self.highlighting_rules.append((r'\*.*?\*', italic_format))
        self.highlighting_rules.append((r'_.*?_', italic_format))
        
        # 程式碼塊
        code_format = QTextCharFormat()
        code_format.setForeground(QColor("#f92672" if is_dark else "#e74c3c"))
        code_format.setBackground(QColor("#3e3e3e" if is_dark else "#f8f9fa"))
        code_format.setFontFamily("Consolas")
        self.highlighting_rules.append((r'`.*?`', code_format))
        
        # 連結
        link_format = QTextCharFormat()
        link_format.setForeground(QColor("#66d9ef" if is_dark else "#3498db"))
        link_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
        self.highlighting_rules.append((r'\[.*?\]\(.*?\)', link_format))
        
        # 引用
        quote_format = QTextCharFormat()
        quote_format.setForeground(QColor("#a6e22e" if is_dark else "#7f8c8d"))
        quote_format.setFontItalic(True)
        self.highlighting_rules.append((r'^>.*', quote_format))
        
        # 列表
        list_format = QTextCharFormat()
        list_format.setForeground(QColor("#fd971f" if is_dark else "#f39c12"))
        self.highlighting_rules.append((r'^\s*[-*+]\s', list_format))
        self.highlighting_rules.append((r'^\s*\d+\.\s', list_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = re.compile(pattern, re.MULTILINE)
            for match in expression.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)


class MarkdownRenderer(QTextBrowser):
    """Markdown渲染器"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setOpenExternalLinks(True)
        self.md = markdown.Markdown(extensions=[
            'codehilite',
            'tables',
            'toc',
            'fenced_code',
            'nl2br'
        ])
        self.apply_theme()
    
    def apply_theme(self):
        """應用主題樣式"""
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_browser_style())
    
    def render_markdown(self, text):
        """渲染Markdown文本"""
        if not text.strip():
            self.setHtml("")
            return
            
        try:
            html = self.md.convert(text)
            css = self.theme_manager.get_markdown_css() if self.theme_manager else ""
            
            # 添加CSS樣式
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    {css}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
            self.setHtml(styled_html)
        except Exception as e:
            self.setPlainText(f"Markdown渲染錯誤: {str(e)}")


class EnhancedTextEdit(QTextEdit):
    """增強的文本編輯器"""
    textChangedDelayed = pyqtSignal(str)
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setFont(QFont("Consolas", 10))
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._emit_delayed_signal)
        self.textChanged.connect(self._on_text_changed)
        
        # 設置製表符寬度
        metrics = self.fontMetrics()
        self.setTabStopDistance(4 * metrics.horizontalAdvance(' '))
        
        self.apply_theme()
        
    def apply_theme(self):
        """應用主題樣式"""
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_editor_style())
    
    def _on_text_changed(self):
        self.timer.stop()
        self.timer.start(300)  # 300ms延遲
        
    def _emit_delayed_signal(self):
        self.textChangedDelayed.emit(self.toPlainText())
    
    def keyPressEvent(self, event):
        # 智能縮排
        if event.key() == Qt.Key.Key_Return:
            cursor = self.textCursor()
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            line = cursor.selectedText()
            indent = len(line) - len(line.lstrip())
            
            super().keyPressEvent(event)
            
            if indent > 0:
                self.insertPlainText(' ' * indent)
        else:
            super().keyPressEvent(event)


class AdvancedSearchDialog(QDialog):
    """高級搜索對話框"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("高級搜索")
        self.setMinimumWidth(450)
        self.setup_ui()
        self.apply_theme()
        
    def apply_theme(self):
        """應用主題樣式"""
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 搜索詞
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索詞:"))
        self.search_input = QLineEdit()
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # 搜索選項
        options_group = QGroupBox("搜索選項")
        options_layout = QVBoxLayout(options_group)
        
        self.case_sensitive = QCheckBox("區分大小寫")
        self.regex_search = QCheckBox("正則表達式")
        self.whole_word = QCheckBox("完整單詞")
        
        options_layout.addWidget(self.case_sensitive)
        options_layout.addWidget(self.regex_search)
        options_layout.addWidget(self.whole_word)
        layout.addWidget(options_group)
        
        # 搜索範圍
        scope_group = QGroupBox("搜索範圍")
        scope_layout = QVBoxLayout(scope_group)
        
        self.radio_prompt = QRadioButton("僅搜索 Prompt")
        self.radio_completion = QRadioButton("僅搜索 Completion")
        self.radio_both = QRadioButton("兩者都搜索")
        self.radio_both.setChecked(True)
        
        scope_layout.addWidget(self.radio_prompt)
        scope_layout.addWidget(self.radio_completion)
        scope_layout.addWidget(self.radio_both)
        layout.addWidget(scope_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        search_button = QPushButton("搜索")
        search_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(search_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def get_search_config(self):
        scope = "both"
        if self.radio_prompt.isChecked():
            scope = "prompt"
        elif self.radio_completion.isChecked():
            scope = "completion"
            
        return {
            'term': self.search_input.text().strip(),
            'scope': scope,
            'case_sensitive': self.case_sensitive.isChecked(),
            'regex': self.regex_search.isChecked(),
            'whole_word': self.whole_word.isChecked()
        }


class DataValidationDialog(QDialog):
    """數據驗證對話框"""
    def __init__(self, issues, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("數據驗證報告")
        self.setMinimumSize(600, 400)
        self.issues = issues
        self.setup_ui()
        self.apply_theme()
        
    def apply_theme(self):
        """應用主題樣式"""
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        if not self.issues:
            layout.addWidget(QLabel("✅ 所有數據項目都通過驗證！"))
        else:
            layout.addWidget(QLabel(f"⚠️ 發現 {len(self.issues)} 個問題："))
            
            issues_list = QListWidget()
            for issue in self.issues:
                issues_list.addItem(f"項目 {issue['index']+1}: {issue['message']}")
            layout.addWidget(issues_list)
        
        close_button = QPushButton("關閉")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class NSJSONReaderEnhanced(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化主題管理器
        self.theme_manager = ThemeManager()
        
        self.setWindowTitle("NS-LLM JSON Reader Enhanced")
        self.resize(1600, 900)
        
        self.current_file = None
        self.data = []
        self.current_index = 0
        self.modified = False
        self.auto_save_enabled = False
        
        self.setup_ui()
        self.create_menu()
        self.create_shortcuts()
        self.apply_theme()
        self.update_ui_state()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 創建工具欄 - 使用QToolBar替代GroupBox
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # 文件操作
        open_btn = QPushButton("📁 開啟")
        open_btn.clicked.connect(self.open_file)
        save_btn = QPushButton("💾 保存")
        save_btn.clicked.connect(self.save_file)
        
        toolbar.addWidget(open_btn)
        toolbar.addWidget(save_btn)
        toolbar.addSeparator()
        
        # 導航控制
        self.prev_button = QPushButton("◀")
        self.prev_button.clicked.connect(self.prev_item)
        self.prev_button.setMaximumWidth(30)
        
        self.next_button = QPushButton("▶")
        self.next_button.clicked.connect(self.next_item)
        self.next_button.setMaximumWidth(30)
        
        self.index_label = QLabel("0/0")
        self.index_label.setMinimumWidth(50)
        
        jump_label = QLabel("跳至:")
        self.jump_input = QLineEdit()
        self.jump_input.setMaximumWidth(50)
        self.jump_input.returnPressed.connect(self.jump_to_index)
        
        toolbar.addWidget(self.prev_button)
        toolbar.addWidget(self.next_button)
        toolbar.addWidget(self.index_label)
        toolbar.addWidget(jump_label)
        toolbar.addWidget(self.jump_input)
        toolbar.addSeparator()
        
        # 編輯操作
        add_btn = QPushButton("➕ 新增")
        add_btn.clicked.connect(self.add_new_item)
        delete_btn = QPushButton("🗑️ 刪除")
        delete_btn.clicked.connect(self.delete_item)
        search_btn = QPushButton("🔍 搜索")
        search_btn.clicked.connect(self.show_advanced_search)
        
        toolbar.addWidget(add_btn)
        toolbar.addWidget(delete_btn)
        toolbar.addWidget(search_btn)
        toolbar.addSeparator()
        
        # 工具
        validate_btn = QPushButton("✓ 驗證")
        validate_btn.clicked.connect(self.validate_data)
        stats_btn = QPushButton("📊 統計")
        stats_btn.clicked.connect(self.show_statistics)
        
        # 主題切換按鈕
        self.theme_btn = QPushButton("🌙 深色" if not self.theme_manager.is_dark_mode else "☀️ 淺色")
        self.theme_btn.clicked.connect(self.toggle_theme)
        
        toolbar.addWidget(validate_btn)
        toolbar.addWidget(stats_btn)
        toolbar.addWidget(self.theme_btn)
        
        # 狀態指示
        toolbar.addSeparator()
        self.modified_label = QLabel("")
        self.modified_label.setStyleSheet("color: red; font-weight: bold; font-size: 9pt;")
        toolbar.addWidget(self.modified_label)
        
        # 將工具欄添加到主視窗
        self.addToolBar(toolbar)
        
        # 主內容區域 - 水平分割（整個界面左右分割）
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側編輯區域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # Prompt區域
        prompt_label = QLabel("📝 Prompt:")
        prompt_label.setFont(QFont("", 11, QFont.Weight.Bold))
        self.prompt_text = EnhancedTextEdit(theme_manager=self.theme_manager)
        self.prompt_text.setMaximumHeight(180)  # 減少Prompt高度
        
        left_layout.addWidget(prompt_label)
        left_layout.addWidget(self.prompt_text)
        
        # Completion編輯區域
        completion_label = QLabel("💬 Completion (編輯):")
        completion_label.setFont(QFont("", 11, QFont.Weight.Bold))
        self.completion_text = EnhancedTextEdit(theme_manager=self.theme_manager)
        self.completion_text.textChangedDelayed.connect(self.on_completion_changed)
        self.completion_highlighter = MarkdownHighlighter(self.completion_text.document(), self.theme_manager)
        
        left_layout.addWidget(completion_label)
        left_layout.addWidget(self.completion_text)
        
        # 右側預覽區域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        preview_label = QLabel("👁️ Completion 實時預覽:")
        preview_label.setFont(QFont("", 11, QFont.Weight.Bold))
        self.completion_renderer = MarkdownRenderer(theme_manager=self.theme_manager)
        
        right_layout.addWidget(preview_label)
        right_layout.addWidget(self.completion_renderer)
        
        # 添加到主分割器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # 設置分割比例：左側編輯區域佔60%，右側預覽區域佔40%
        main_splitter.setSizes([960, 640])  # 基於1600寬度的比例
        
        main_layout.addWidget(main_splitter)
        
        # 狀態欄
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就緒")
        
        # 連接文本變化信號
        self.prompt_text.textChanged.connect(lambda: self.set_modified(True))
        self.completion_text.textChanged.connect(lambda: self.set_modified(True))
    
    def toggle_theme(self):
        """切換主題"""
        self.theme_manager.toggle_theme()
        self.apply_theme()
        
        # 更新按鈕文字
        self.theme_btn.setText("🌙 深色" if not self.theme_manager.is_dark_mode else "☀️ 淺色")
        
        # 重新渲染Markdown
        if hasattr(self, 'completion_text'):
            self.on_completion_changed(self.completion_text.toPlainText())
    
    def apply_theme(self):
        """應用主題到所有組件"""
        # 應用應用程式整體樣式
        self.setStyleSheet(self.theme_manager.get_app_style())
        
        # 更新各個組件的主題
        if hasattr(self, 'prompt_text'):
            self.prompt_text.apply_theme()
            
        if hasattr(self, 'completion_text'):
            self.completion_text.apply_theme()
            self.completion_highlighter.update_highlighting_rules()
            self.completion_renderer.apply_theme()
    
    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜單
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("開啟 (Ctrl+O)", self.open_file)
        file_menu.addAction("保存 (Ctrl+S)", self.save_file)
        file_menu.addAction("另存為 (Ctrl+Shift+S)", self.save_file_as)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)
        
        # 編輯菜單
        edit_menu = menubar.addMenu("編輯")
        edit_menu.addAction("搜索 (Ctrl+F)", self.show_advanced_search)
        edit_menu.addAction("新增項目", self.add_new_item)
        edit_menu.addAction("刪除項目", self.delete_item)
        edit_menu.addAction("應用更改 (Ctrl+Enter)", self.apply_changes)
        
        # 工具菜單
        tools_menu = menubar.addMenu("工具")
        tools_menu.addAction("數據驗證", self.validate_data)
        tools_menu.addAction("統計信息", self.show_statistics)
        
        # 視圖菜單
        view_menu = menubar.addMenu("視圖")
        auto_save_action = view_menu.addAction("自動保存")
        auto_save_action.setCheckable(True)
        auto_save_action.toggled.connect(self.toggle_auto_save)
        
        theme_action = view_menu.addAction("切換主題 (Ctrl+T)")
        theme_action.triggered.connect(self.toggle_theme)
    
    def create_shortcuts(self):
        shortcuts = [
            ("Ctrl+O", self.open_file),
            ("Ctrl+S", self.save_file),
            ("Ctrl+Shift+S", self.save_file_as),
            ("Ctrl+F", self.show_advanced_search),
            ("Ctrl+Return", self.apply_changes),
            ("Ctrl+Left", self.prev_item),
            ("Ctrl+Right", self.next_item),
            ("Ctrl+T", self.toggle_theme),
        ]
        
        for shortcut, slot in shortcuts:
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            self.addAction(action)
    
    def on_completion_changed(self, text):
        """Completion文本改變時的處理"""
        self.completion_renderer.render_markdown(text)
    
    def set_modified(self, value):
        self.modified = value
        self.modified_label.setText("* 已修改" if value else "")
    
    def update_ui_state(self):
        has_data = len(self.data) > 0
        
        self.prev_button.setEnabled(has_data and self.current_index > 0)
        self.next_button.setEnabled(has_data and self.current_index < len(self.data) - 1)
        self.prompt_text.setEnabled(has_data)
        self.completion_text.setEnabled(has_data)
        self.jump_input.setEnabled(has_data)
        
        if has_data:
            self.index_label.setText(f"{self.current_index + 1}/{len(self.data)}")
        else:
            self.index_label.setText("0/0")
    
    def open_file(self):
        if self.check_unsaved_changes():
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟文件", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                
                self.current_file = file_path
                self.current_index = 0
                self.set_modified(False)
                
                self.update_display()
                self.update_ui_state()
                
                self.setWindowTitle(f"NS-LLM JSON Reader Enhanced - {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"已開啟: {file_path} | {len(self.data)} 個項目")
                
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法開啟文件: {str(e)}")
    
    def save_file(self):
        if not self.current_file:
            return self.save_file_as()
        
        try:
            if self.modified:
                self.apply_changes_silent()
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            self.set_modified(False)
            self.statusBar.showMessage(f"已保存: {self.current_file}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法保存文件: {str(e)}")
            return False
    
    def save_file_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "另存為", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.current_file = file_path
            return self.save_file()
        return False
    
    def update_display(self):
        if not self.data:
            self.prompt_text.clear()
            self.completion_text.clear()
            self.completion_renderer.setHtml("")
            return
        
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.data):
            self.current_index = len(self.data) - 1
        
        current_item = self.data[self.current_index]
        
        # 阻止信號
        self.prompt_text.blockSignals(True)
        self.completion_text.blockSignals(True)
        
        prompt_text = current_item.get("prompt", "")
        completion_text = current_item.get("completion", "")
        
        self.prompt_text.setPlainText(prompt_text)
        self.completion_text.setPlainText(completion_text)
        
        # 更新預覽
        self.completion_renderer.render_markdown(completion_text)
        
        self.prompt_text.blockSignals(False)
        self.completion_text.blockSignals(False)
        
        self.set_modified(False)
        self.update_ui_state()
    
    def apply_changes_silent(self):
        if not self.data:
            return
        
        self.data[self.current_index]["prompt"] = self.prompt_text.toPlainText().strip()
        self.data[self.current_index]["completion"] = self.completion_text.toPlainText().strip()
        self.set_modified(False)
    
    def apply_changes(self):
        self.apply_changes_silent()
        QMessageBox.information(self, "成功", "已應用更改！")
    
    def check_unsaved_changes(self):
        if self.modified:
            reply = QMessageBox.question(
                self, "未保存的更改", "是否保存當前更改？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            
            if reply == QMessageBox.StandardButton.Yes:
                return not self.save_file()
            elif reply == QMessageBox.StandardButton.Cancel:
                return True
        return False
    
    def prev_item(self):
        if self.current_index > 0:
            if self.check_unsaved_changes():
                return
            self.current_index -= 1
            self.update_display()
    
    def next_item(self):
        if self.current_index < len(self.data) - 1:
            if self.check_unsaved_changes():
                return
            self.current_index += 1
            self.update_display()
    
    def jump_to_index(self):
        try:
            index = int(self.jump_input.text()) - 1
            if 0 <= index < len(self.data):
                if self.check_unsaved_changes():
                    return
                self.current_index = index
                self.update_display()
            else:
                QMessageBox.warning(self, "錯誤", f"索引必須在 1 和 {len(self.data)} 之間")
        except ValueError:
            QMessageBox.warning(self, "錯誤", "請輸入有效數字")
    
    def add_new_item(self):
        if self.check_unsaved_changes():
            return
        
        new_item = {"prompt": "", "completion": ""}
        self.data.append(new_item)
        self.current_index = len(self.data) - 1
        self.update_display()
        self.set_modified(True)
    
    def delete_item(self):
        if not self.data:
            return
        
        reply = QMessageBox.question(
            self, "確認刪除", "確定要刪除此項目？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.data[self.current_index]
            if self.current_index >= len(self.data) and self.data:
                self.current_index = len(self.data) - 1
            self.update_display()
            self.set_modified(True)
    
    def show_advanced_search(self):
        dialog = AdvancedSearchDialog(self, self.theme_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_search_config()
            if config['term']:
                self.search_with_config(config)
    
    def search_with_config(self, config):
        found = []
        term = config['term']
        
        for i, item in enumerate(self.data):
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            
            # 根據配置調整搜索
            if not config['case_sensitive']:
                term_search = term.lower()
                prompt_search = prompt.lower()
                completion_search = completion.lower()
            else:
                term_search = term
                prompt_search = prompt
                completion_search = completion
            
            match_found = False
            
            if config['scope'] in ['prompt', 'both']:
                if config['regex']:
                    try:
                        if re.search(term_search, prompt_search):
                            match_found = True
                    except re.error:
                        pass
                else:
                    if term_search in prompt_search:
                        match_found = True
            
            if not match_found and config['scope'] in ['completion', 'both']:
                if config['regex']:
                    try:
                        if re.search(term_search, completion_search):
                            match_found = True
                    except re.error:
                        pass
                else:
                    if term_search in completion_search:
                        match_found = True
            
            if match_found:
                preview = prompt[:50].replace('\n', ' ')
                if len(prompt) > 50:
                    preview += "..."
                found.append((i, preview))
        
        if found:
            self.show_search_results(term, found)
        else:
            QMessageBox.information(self, "搜索", f"未找到 '{term}' 的匹配項")
    
    def show_search_results(self, term, results):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"搜索結果: '{term}'")
        dialog.setMinimumSize(500, 400)
        
        # 應用主題
        dialog.setStyleSheet(self.theme_manager.get_app_style())
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"找到 {len(results)} 個匹配項:"))
        
        result_list = QListWidget()
        for idx, preview in results:
            result_list.addItem(f"{idx+1}: {preview}")
        layout.addWidget(result_list)
        
        button_layout = QHBoxLayout()
        go_button = QPushButton("前往")
        go_button.clicked.connect(lambda: self.go_to_result(dialog, result_list, results))
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(go_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        result_list.itemDoubleClicked.connect(lambda: self.go_to_result(dialog, result_list, results))
        dialog.exec()
    
    def go_to_result(self, dialog, result_list, results):
        current_row = result_list.currentRow()
        if current_row >= 0:
            if self.check_unsaved_changes():
                return
            self.current_index = results[current_row][0]
            self.update_display()
            dialog.accept()
    
    def validate_data(self):
        issues = []
        
        for i, item in enumerate(self.data):
            if not isinstance(item, dict):
                issues.append({'index': i, 'message': '項目不是字典類型'})
                continue
            
            if 'prompt' not in item:
                issues.append({'index': i, 'message': '缺少 prompt 字段'})
            elif not item['prompt'].strip():
                issues.append({'index': i, 'message': 'prompt 為空'})
            
            if 'completion' not in item:
                issues.append({'index': i, 'message': '缺少 completion 字段'})
            elif not item['completion'].strip():
                issues.append({'index': i, 'message': 'completion 為空'})
        
        dialog = DataValidationDialog(issues, self, self.theme_manager)
        dialog.exec()
    
    def show_statistics(self):
        if not self.data:
            QMessageBox.information(self, "統計", "沒有數據")
            return
        
        total_items = len(self.data)
        total_prompt_chars = sum(len(item.get('prompt', '')) for item in self.data)
        total_completion_chars = sum(len(item.get('completion', '')) for item in self.data)
        avg_prompt_len = total_prompt_chars / total_items
        avg_completion_len = total_completion_chars / total_items
        
        stats_text = f"""
統計信息:
- 總項目數: {total_items}
- Prompt總字符數: {total_prompt_chars:,}
- Completion總字符數: {total_completion_chars:,}
- 平均Prompt長度: {avg_prompt_len:.1f}
- 平均Completion長度: {avg_completion_len:.1f}
- 總字符數: {total_prompt_chars + total_completion_chars:,}
        """
        
        QMessageBox.information(self, "統計信息", stats_text)
    
    def toggle_auto_save(self, enabled):
        self.auto_save_enabled = enabled
        if enabled:
            self.statusBar.showMessage("自動保存已啟用")
        else:
            self.statusBar.showMessage("自動保存已禁用")
    
    def closeEvent(self, event):
        if self.check_unsaved_changes():
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 安裝markdown如果尚未安裝
    try:
        import markdown
    except ImportError:
        QMessageBox.critical(None, "缺少依賴", "請安裝markdown庫: pip install markdown")
        sys.exit(1)
    
    window = NSJSONReaderEnhanced()
    window.show()
    sys.exit(app.exec())