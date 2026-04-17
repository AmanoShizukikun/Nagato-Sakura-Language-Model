import sys
import json
import os
import re
import csv
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
                    margin: 15px 0;
                    border: 2px solid #555;
                }
                th, td { 
                    border: 1px solid #555; 
                    padding: 8px 12px; 
                    text-align: left; 
                    vertical-align: top;
                }
                th { 
                    background-color: #404040; 
                    color: #ffffff;
                    font-weight: bold;
                }
                td {
                    background-color: #2b2b2b;
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
                    margin: 15px 0;
                    border: 2px solid #dee2e6;
                }
                th, td { 
                    border: 1px solid #dee2e6; 
                    padding: 8px 12px; 
                    text-align: left; 
                    vertical-align: top;
                }
                th { 
                    background-color: #f8f9fa; 
                    color: #495057;
                    font-weight: bold;
                }
                td {
                    background-color: #ffffff;
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
            # 重置markdown解析器狀態
            self.md.reset()
            
            # 預處理文本，修復表格顯示問題
            processed_text = self.preprocess_markdown(text)
            
            html = self.md.convert(processed_text)
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
    
    def preprocess_markdown(self, text):
        """預處理Markdown文本，修復表格顯示問題"""
        lines = text.split('\n')
        processed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # 檢測表格開始：包含 | 且下一行是分隔符
            if self.is_table_header(line, lines, i):
                # 處理整個表格
                table_lines = []
                j = i
                
                # 添加表格頭
                table_lines.append(lines[j])
                j += 1
                
                # 添加分隔符行
                if j < len(lines):
                    table_lines.append(lines[j])
                    j += 1
                
                # 添加表格內容行
                while j < len(lines) and self.is_table_row(lines[j]):
                    table_lines.append(lines[j])
                    j += 1
                
                # 確保表格前後有空行以正確分離內容
                if processed_lines and processed_lines[-1].strip():
                    processed_lines.append('')
                
                processed_lines.extend(table_lines)
                
                # 表格後添加兩個空行確保完全分離
                processed_lines.append('')
                processed_lines.append('')
                
                i = j
                continue
            
            processed_lines.append(line)
            i += 1
        
        # 清理多餘的空行（超過兩個連續空行的情況）
        final_lines = []
        empty_count = 0
        
        for line in processed_lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:  # 最多保留兩個空行
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def is_table_header(self, line, lines, index):
        """檢查是否是表格頭行"""
        if '|' not in line or not line.strip():
            return False
        
        # 檢查下一行是否是表格分隔符
        if index + 1 >= len(lines):
            return False
        
        next_line = lines[index + 1].strip()
        # 表格分隔符應該包含 | 和 - 和可選的 :
        # 更嚴格的正則表達式
        return bool(re.match(r'^\s*\|[\s\-:|]+\|\s*$', next_line))
    
    def is_table_row(self, line):
        """檢查是否是表格行"""
        line = line.strip()
        if not line:
            return False
        
        # 表格行應該包含 | 但不是分隔符行
        has_pipe = '|' in line
        is_separator = bool(re.match(r'^\s*\|[\s\-:|]+\|\s*$', line))
        
        return has_pipe and not is_separator


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


class BatchOperationsDialog(QDialog):
    """批量操作對話框"""
    def __init__(self, data, parent=None, theme_manager=None):
        super().__init__(parent)
        self.data = data
        self.theme_manager = theme_manager
        self.operations_performed = False
        self.setWindowTitle("批量操作")
        self.setMinimumSize(500, 400)
        self.setup_ui()
        self.apply_theme()
        
    def apply_theme(self):
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 操作類型選擇
        operations_group = QGroupBox("選擇操作類型")
        operations_layout = QVBoxLayout(operations_group)
        
        self.replace_text_radio = QRadioButton("批量替換文本")
        self.add_prefix_radio = QRadioButton("添加前綴")
        self.add_suffix_radio = QRadioButton("添加後綴")
        self.remove_empty_radio = QRadioButton("移除空項目")
        self.normalize_whitespace_radio = QRadioButton("標準化空白字符")
        
        self.replace_text_radio.setChecked(True)
        
        operations_layout.addWidget(self.replace_text_radio)
        operations_layout.addWidget(self.add_prefix_radio)
        operations_layout.addWidget(self.add_suffix_radio)
        operations_layout.addWidget(self.remove_empty_radio)
        operations_layout.addWidget(self.normalize_whitespace_radio)
        layout.addWidget(operations_group)
        
        # 參數輸入
        params_group = QGroupBox("操作參數")
        params_layout = QVBoxLayout(params_group)
        
        # 查找替換
        find_layout = QHBoxLayout()
        find_layout.addWidget(QLabel("查找:"))
        self.find_input = QLineEdit()
        find_layout.addWidget(self.find_input)
        params_layout.addLayout(find_layout)
        
        replace_layout = QHBoxLayout()
        replace_layout.addWidget(QLabel("替換:"))
        self.replace_input = QLineEdit()
        replace_layout.addWidget(self.replace_input)
        params_layout.addLayout(replace_layout)
        
        # 前綴/後綴
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("前綴:"))
        self.prefix_input = QLineEdit()
        prefix_layout.addWidget(self.prefix_input)
        params_layout.addLayout(prefix_layout)
        
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("後綴:"))
        self.suffix_input = QLineEdit()
        suffix_layout.addWidget(self.suffix_input)
        params_layout.addLayout(suffix_layout)
        
        layout.addWidget(params_group)
        
        # 目標選擇
        target_group = QGroupBox("應用到")
        target_layout = QVBoxLayout(target_group)
        
        self.target_prompt = QCheckBox("Prompt")
        self.target_completion = QCheckBox("Completion")
        self.target_both = QCheckBox("兩者")
        self.target_both.setChecked(True)
        
        target_layout.addWidget(self.target_prompt)
        target_layout.addWidget(self.target_completion)
        target_layout.addWidget(self.target_both)
        layout.addWidget(target_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        execute_btn = QPushButton("執行")
        execute_btn.clicked.connect(self.execute_operation)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(execute_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def execute_operation(self):
        """執行批量操作"""
        try:
            affected_count = 0
            
            if self.replace_text_radio.isChecked():
                find_text = self.find_input.text()
                replace_text = self.replace_input.text()
                if not find_text:
                    QMessageBox.warning(self, "錯誤", "請輸入要查找的文本")
                    return
                affected_count = self.replace_text(find_text, replace_text)
                
            elif self.add_prefix_radio.isChecked():
                prefix = self.prefix_input.text()
                if not prefix:
                    QMessageBox.warning(self, "錯誤", "請輸入前綴")
                    return
                affected_count = self.add_prefix(prefix)
                
            elif self.add_suffix_radio.isChecked():
                suffix = self.suffix_input.text()
                if not suffix:
                    QMessageBox.warning(self, "錯誤", "請輸入後綴")
                    return
                affected_count = self.add_suffix(suffix)
                
            elif self.remove_empty_radio.isChecked():
                affected_count = self.remove_empty_items()
                
            elif self.normalize_whitespace_radio.isChecked():
                affected_count = self.normalize_whitespace()
            
            self.operations_performed = True
            QMessageBox.information(self, "完成", f"操作完成！影響了 {affected_count} 個項目。")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"操作失敗: {str(e)}")
    
    def get_targets(self):
        """獲取操作目標"""
        targets = []
        if self.target_both.isChecked():
            targets = ['prompt', 'completion']
        else:
            if self.target_prompt.isChecked():
                targets.append('prompt')
            if self.target_completion.isChecked():
                targets.append('completion')
        return targets
    
    def replace_text(self, find_text, replace_text):
        """批量替換文本"""
        affected = 0
        targets = self.get_targets()
        
        for item in self.data:
            for target in targets:
                if target in item and find_text in item[target]:
                    item[target] = item[target].replace(find_text, replace_text)
                    affected += 1
        return affected
    
    def add_prefix(self, prefix):
        """添加前綴"""
        affected = 0
        targets = self.get_targets()
        
        for item in self.data:
            for target in targets:
                if target in item and item[target].strip():
                    if not item[target].startswith(prefix):
                        item[target] = prefix + item[target]
                        affected += 1
        return affected
    
    def add_suffix(self, suffix):
        """添加後綴"""
        affected = 0
        targets = self.get_targets()
        
        for item in self.data:
            for target in targets:
                if target in item and item[target].strip():
                    if not item[target].endswith(suffix):
                        item[target] = item[target] + suffix
                        affected += 1
        return affected
    
    def remove_empty_items(self):
        """移除空項目"""
        original_count = len(self.data)
        self.data[:] = [item for item in self.data 
                       if item.get('prompt', '').strip() and item.get('completion', '').strip()]
        return original_count - len(self.data)
    
    def normalize_whitespace(self):
        """標準化空白字符"""
        affected = 0
        targets = self.get_targets()
        
        for item in self.data:
            for target in targets:
                if target in item:
                    original = item[target]
                    # 移除多餘空白、統一換行符
                    normalized = re.sub(r'\s+', ' ', original.strip())
                    normalized = normalized.replace(' \n ', '\n').replace('\n ', '\n')
                    if original != normalized:
                        item[target] = normalized
                        affected += 1
        return affected


class DataCleaningDialog(QDialog):
    """數據清理對話框"""
    def __init__(self, data, parent=None, theme_manager=None):
        super().__init__(parent)
        self.data = data
        self.theme_manager = theme_manager
        self.cleaning_performed = False
        self.setWindowTitle("數據清理")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        self.apply_theme()
        
    def apply_theme(self):
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 清理選項
        options_group = QGroupBox("清理選項")
        options_layout = QVBoxLayout(options_group)
        
        self.remove_empty_lines = QCheckBox("移除空行")
        self.remove_extra_spaces = QCheckBox("移除多餘空格")
        self.fix_encoding = QCheckBox("修復編碼問題")
        self.remove_duplicates = QCheckBox("移除重複項目")
        self.fix_quotes = QCheckBox("統一引號格式")
        self.remove_special_chars = QCheckBox("移除特殊控制字符")
        self.standardize_newlines = QCheckBox("標準化換行符")
        
        self.remove_empty_lines.setChecked(True)
        self.remove_extra_spaces.setChecked(True)
        self.standardize_newlines.setChecked(True)
        
        options_layout.addWidget(self.remove_empty_lines)
        options_layout.addWidget(self.remove_extra_spaces)
        options_layout.addWidget(self.fix_encoding)
        options_layout.addWidget(self.remove_duplicates)
        options_layout.addWidget(self.fix_quotes)
        options_layout.addWidget(self.remove_special_chars)
        options_layout.addWidget(self.standardize_newlines)
        layout.addWidget(options_group)
        
        # 預覽區域
        preview_group = QGroupBox("清理預覽")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextBrowser()
        self.preview_text.setMaximumHeight(200)
        self.preview_text.setStyleSheet(self.theme_manager.get_browser_style() if self.theme_manager else "")
        preview_layout.addWidget(self.preview_text)
        
        preview_btn = QPushButton("預覽清理效果")
        preview_btn.clicked.connect(self.preview_cleaning)
        preview_layout.addWidget(preview_btn)
        
        layout.addWidget(preview_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        execute_btn = QPushButton("開始清理")
        execute_btn.clicked.connect(self.execute_cleaning)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(execute_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def preview_cleaning(self):
        """預覽清理效果"""
        if not self.data:
            return
            
        # 取前幾個項目進行預覽
        sample_data = self.data[:3]
        preview_results = []
        
        for i, item in enumerate(sample_data):
            original_prompt = item.get('prompt', '')
            original_completion = item.get('completion', '')
            
            cleaned_prompt = self.clean_text(original_prompt)
            cleaned_completion = self.clean_text(original_completion)
            
            preview_results.append(f"""
項目 {i+1}:
原始 Prompt: {original_prompt[:100]}{'...' if len(original_prompt) > 100 else ''}
清理後 Prompt: {cleaned_prompt[:100]}{'...' if len(cleaned_prompt) > 100 else ''}

原始 Completion: {original_completion[:100]}{'...' if len(original_completion) > 100 else ''}
清理後 Completion: {cleaned_completion[:100]}{'...' if len(cleaned_completion) > 100 else ''}
{'='*50}
""")
        
        self.preview_text.setPlainText('\n'.join(preview_results))
    
    def clean_text(self, text):
        """根據選擇的選項清理文本"""
        if not text:
            return text
            
        cleaned = text
        
        if self.remove_empty_lines.isChecked():
            cleaned = '\n'.join(line for line in cleaned.split('\n') if line.strip())
        
        if self.remove_extra_spaces.isChecked():
            cleaned = re.sub(r' +', ' ', cleaned)
            cleaned = re.sub(r'\n +', '\n', cleaned)
            cleaned = re.sub(r' +\n', '\n', cleaned)
        
        if self.fix_encoding.isChecked():
            # 修復常見的編碼問題
            replacements = {
                'â€™': "'", 'â€œ': '"', 'â€': '"',
                'â€¦': '...', 'â€"': '—', 'â€"': '–'
            }
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
        
        if self.fix_quotes.isChecked():
            # 統一引號格式
            cleaned = cleaned.replace('"', '"').replace('"', '"')
            cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        if self.remove_special_chars.isChecked():
            # 移除控制字符，但保留常用的
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        if self.standardize_newlines.isChecked():
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            # 移除多餘的換行符
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def execute_cleaning(self):
        """執行數據清理"""
        try:
            affected_count = 0
            duplicates_removed = 0
            
            # 清理文本
            for item in self.data:
                original_prompt = item.get('prompt', '')
                original_completion = item.get('completion', '')
                
                cleaned_prompt = self.clean_text(original_prompt)
                cleaned_completion = self.clean_text(original_completion)
                
                if original_prompt != cleaned_prompt or original_completion != cleaned_completion:
                    item['prompt'] = cleaned_prompt
                    item['completion'] = cleaned_completion
                    affected_count += 1
            
            # 移除重複項目
            if self.remove_duplicates.isChecked():
                seen = set()
                unique_data = []
                for item in self.data:
                    key = (item.get('prompt', ''), item.get('completion', ''))
                    if key not in seen:
                        seen.add(key)
                        unique_data.append(item)
                    else:
                        duplicates_removed += 1
                
                self.data[:] = unique_data
            
            self.cleaning_performed = True
            message = f"清理完成！\n清理了 {affected_count} 個項目"
            if duplicates_removed > 0:
                message += f"\n移除了 {duplicates_removed} 個重複項目"
            
            QMessageBox.information(self, "完成", message)
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"清理失敗: {str(e)}")


class ExportDialog(QDialog):
    """導出對話框"""
    def __init__(self, data, parent=None, theme_manager=None):
        super().__init__(parent)
        self.data = data
        self.theme_manager = theme_manager
        self.setWindowTitle("導出數據")
        self.setMinimumSize(400, 300)
        self.setup_ui()
        self.apply_theme()
        
    def apply_theme(self):
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 導出格式選擇
        format_group = QGroupBox("導出格式")
        format_layout = QVBoxLayout(format_group)
        
        self.json_radio = QRadioButton("JSON (訓練格式)")
        self.jsonl_radio = QRadioButton("JSONL (逐行JSON)")
        self.csv_radio = QRadioButton("CSV")
        self.txt_radio = QRadioButton("純文本")
        self.markdown_radio = QRadioButton("Markdown")
        
        self.json_radio.setChecked(True)
        
        format_layout.addWidget(self.json_radio)
        format_layout.addWidget(self.jsonl_radio)
        format_layout.addWidget(self.csv_radio)
        format_layout.addWidget(self.txt_radio)
        format_layout.addWidget(self.markdown_radio)
        layout.addWidget(format_group)
        
        # 範圍選擇
        range_group = QGroupBox("導出範圍")
        range_layout = QVBoxLayout(range_group)
        
        self.all_radio = QRadioButton("全部數據")
        self.range_radio = QRadioButton("指定範圍")
        self.all_radio.setChecked(True)
        
        range_layout.addWidget(self.all_radio)
        range_layout.addWidget(self.range_radio)
        
        # 範圍輸入
        range_input_layout = QHBoxLayout()
        range_input_layout.addWidget(QLabel("從:"))
        self.start_input = QSpinBox()
        self.start_input.setMinimum(1)
        self.start_input.setMaximum(len(self.data))
        self.start_input.setValue(1)
        range_input_layout.addWidget(self.start_input)
        
        range_input_layout.addWidget(QLabel("到:"))
        self.end_input = QSpinBox()
        self.end_input.setMinimum(1)
        self.end_input.setMaximum(len(self.data))
        self.end_input.setValue(len(self.data))
        range_input_layout.addWidget(self.end_input)
        
        range_layout.addLayout(range_input_layout)
        layout.addWidget(range_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        export_btn = QPushButton("導出")
        export_btn.clicked.connect(self.export_data)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(export_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def export_data(self):
        """執行導出"""
        try:
            # 獲取導出數據
            if self.all_radio.isChecked():
                export_data = self.data
            else:
                start = self.start_input.value() - 1
                end = self.end_input.value()
                export_data = self.data[start:end]
            
            # 確定文件擴展名
            if self.json_radio.isChecked():
                ext = "json"
                filter_str = "JSON Files (*.json)"
            elif self.jsonl_radio.isChecked():
                ext = "jsonl"
                filter_str = "JSONL Files (*.jsonl)"
            elif self.csv_radio.isChecked():
                ext = "csv"
                filter_str = "CSV Files (*.csv)"
            elif self.txt_radio.isChecked():
                ext = "txt"
                filter_str = "Text Files (*.txt)"
            else:  # markdown
                ext = "md"
                filter_str = "Markdown Files (*.md)"
            
            # 選擇保存位置
            file_path, _ = QFileDialog.getSaveFileName(
                self, "導出到...", f"exported_data.{ext}", filter_str
            )
            
            if not file_path:
                return
            
            # 執行導出
            if self.json_radio.isChecked():
                self.export_json(export_data, file_path)
            elif self.jsonl_radio.isChecked():
                self.export_jsonl(export_data, file_path)
            elif self.csv_radio.isChecked():
                self.export_csv(export_data, file_path)
            elif self.txt_radio.isChecked():
                self.export_txt(export_data, file_path)
            else:
                self.export_markdown(export_data, file_path)
            
            QMessageBox.information(self, "成功", f"數據已導出到: {file_path}")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"導出失敗: {str(e)}")
    
    def export_json(self, data, file_path):
        """導出為JSON格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def export_jsonl(self, data, file_path):
        """導出為JSONL格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def export_csv(self, data, file_path):
        """導出為CSV格式"""
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'completion'])
            for item in data:
                writer.writerow([item.get('prompt', ''), item.get('completion', '')])
    
    def export_txt(self, data, file_path):
        """導出為純文本格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                f.write(f"項目 {i+1}:\n")
                f.write(f"Prompt: {item.get('prompt', '')}\n")
                f.write(f"Completion: {item.get('completion', '')}\n")
                f.write("-" * 50 + "\n\n")
    
    def export_markdown(self, data, file_path):
        """導出為Markdown格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# 訓練數據導出\n\n")
            for i, item in enumerate(data):
                f.write(f"## 項目 {i+1}\n\n")
                f.write(f"**Prompt:**\n{item.get('prompt', '')}\n\n")
                f.write(f"**Completion:**\n{item.get('completion', '')}\n\n")
                f.write("---\n\n")


class CharacterAnalysisDialog(QDialog):
    """字符分析對話框"""
    def __init__(self, data, parent=None, theme_manager=None):
        super().__init__(parent)
        self.data = data
        self.theme_manager = theme_manager
        self.setWindowTitle("字符分析")
        self.setMinimumSize(700, 600)
        self.setup_ui()
        self.apply_theme()
        self.perform_analysis()
        
    def apply_theme(self):
        if self.theme_manager:
            self.setStyleSheet(self.theme_manager.get_app_style())
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 分析結果顯示
        self.analysis_text = QTextBrowser()
        self.analysis_text.setStyleSheet(self.theme_manager.get_browser_style() if self.theme_manager else "")
        layout.addWidget(self.analysis_text)
        
        # 按鈕
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("重新分析")
        refresh_btn.clicked.connect(self.perform_analysis)
        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
    
    def perform_analysis(self):
        """執行字符分析"""
        if not self.data:
            self.analysis_text.setPlainText("沒有數據可分析")
            return
        
        analysis_results = []
        
        # 基本統計
        total_items = len(self.data)
        all_text = ""
        prompt_texts = []
        completion_texts = []
        
        for item in self.data:
            prompt = item.get('prompt', '')
            completion = item.get('completion', '')
            prompt_texts.append(prompt)
            completion_texts.append(completion)
            all_text += prompt + " " + completion + " "
        
        # 字符統計
        char_count = {}
        for char in all_text:
            char_count[char] = char_count.get(char, 0) + 1
        
        # 最常見的字符
        most_common_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # 語言檢測（簡單的中英文比例）
        chinese_chars = sum(1 for char in all_text if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in all_text if char.isalpha() and char.isascii())
        
        # 特殊字符統計
        special_chars = sum(1 for char in all_text if not char.isalnum() and not char.isspace())
        
        # Emoji統計
        emoji_count = sum(1 for char in all_text if ord(char) > 0x1F600)
        
        # 詞彙豐富度（簡單的唯一詞計算）
        words = all_text.split()
        unique_words = len(set(words))
        total_words = len(words)
        
        analysis_results.append("# 字符分析報告\n")
        analysis_results.append(f"## 基本統計")
        analysis_results.append(f"- 總項目數: {total_items:,}")
        analysis_results.append(f"- 總字符數: {len(all_text):,}")
        analysis_results.append(f"- 總詞數: {total_words:,}")
        analysis_results.append(f"- 唯一詞數: {unique_words:,}")
        analysis_results.append(f"- 詞彙豐富度: {(unique_words/total_words*100):.1f}%\n")
        
        analysis_results.append(f"## 語言分布")
        analysis_results.append(f"- 中文字符: {chinese_chars:,} ({chinese_chars/len(all_text)*100:.1f}%)")
        analysis_results.append(f"- 英文字符: {english_chars:,} ({english_chars/len(all_text)*100:.1f}%)")
        analysis_results.append(f"- 特殊字符: {special_chars:,} ({special_chars/len(all_text)*100:.1f}%)")
        analysis_results.append(f"- Emoji/符號: {emoji_count:,}\n")
        
        analysis_results.append(f"## 最常見字符 (前20)")
        for i, (char, count) in enumerate(most_common_chars[:20]):
            char_display = char if char.isprintable() and char != ' ' else repr(char)
            analysis_results.append(f"{i+1:2d}. {char_display}: {count:,}")
        
        # 長度分析
        prompt_lengths = [len(text) for text in prompt_texts]
        completion_lengths = [len(text) for text in completion_texts]
        
        analysis_results.append(f"\n## 長度分析")
        analysis_results.append(f"### Prompt 長度")
        analysis_results.append(f"- 平均: {sum(prompt_lengths)/len(prompt_lengths):.1f}")
        analysis_results.append(f"- 最短: {min(prompt_lengths)}")
        analysis_results.append(f"- 最長: {max(prompt_lengths)}")
        
        analysis_results.append(f"### Completion 長度")
        analysis_results.append(f"- 平均: {sum(completion_lengths)/len(completion_lengths):.1f}")
        analysis_results.append(f"- 最短: {min(completion_lengths)}")
        analysis_results.append(f"- 最長: {max(completion_lengths)}")
        
        self.analysis_text.setPlainText('\n'.join(analysis_results))


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
        tools_menu.addAction("批量操作", self.show_batch_operations)
        tools_menu.addAction("數據清理", self.show_data_cleaning)
        tools_menu.addAction("導出數據", self.export_data)
        tools_menu.addAction("字符分析", self.show_character_analysis)
        
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
        
        # 長度分布統計
        prompt_lengths = [len(item.get('prompt', '')) for item in self.data]
        completion_lengths = [len(item.get('completion', '')) for item in self.data]
        
        prompt_min, prompt_max = min(prompt_lengths), max(prompt_lengths)
        completion_min, completion_max = min(completion_lengths), max(completion_lengths)
        
        # 空值統計
        empty_prompts = sum(1 for item in self.data if not item.get('prompt', '').strip())
        empty_completions = sum(1 for item in self.data if not item.get('completion', '').strip())
        
        stats_text = f"""
詳細統計信息:
=====================================
基本統計:
- 總項目數: {total_items:,}
- 有效項目數: {total_items - max(empty_prompts, empty_completions):,}

Prompt統計:
- 總字符數: {total_prompt_chars:,}
- 平均長度: {avg_prompt_len:.1f}
- 最短長度: {prompt_min}
- 最長長度: {prompt_max}
- 空值數量: {empty_prompts}

Completion統計:
- 總字符數: {total_completion_chars:,}
- 平均長度: {avg_completion_len:.1f}
- 最短長度: {completion_min}
- 最長長度: {completion_max}
- 空值數量: {empty_completions}

總體統計:
- 總字符數: {total_prompt_chars + total_completion_chars:,}
- 數據完整性: {((total_items - max(empty_prompts, empty_completions)) / total_items * 100):.1f}%
        """
        
        # 創建統計窗口
        stats_dialog = QDialog(self)
        stats_dialog.setWindowTitle("詳細統計信息")
        stats_dialog.setMinimumSize(500, 600)
        stats_dialog.setStyleSheet(self.theme_manager.get_app_style())
        
        layout = QVBoxLayout(stats_dialog)
        
        # 統計文本
        stats_browser = QTextBrowser()
        stats_browser.setPlainText(stats_text)
        stats_browser.setStyleSheet(self.theme_manager.get_browser_style())
        layout.addWidget(stats_browser)
        
        # 關閉按鈕
        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(stats_dialog.accept)
        layout.addWidget(close_btn)
        
        stats_dialog.exec()
    
    def show_batch_operations(self):
        """顯示批量操作對話框"""
        dialog = BatchOperationsDialog(self.data, self, self.theme_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.operations_performed:
                self.set_modified(True)
                self.update_display()
                QMessageBox.information(self, "成功", "批量操作已完成！")
    
    def show_data_cleaning(self):
        """顯示數據清理對話框"""
        dialog = DataCleaningDialog(self.data, self, self.theme_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.cleaning_performed:
                self.set_modified(True)
                self.update_display()
                QMessageBox.information(self, "成功", "數據清理已完成！")
    
    def export_data(self):
        """導出數據到不同格式"""
        if not self.data:
            QMessageBox.information(self, "導出", "沒有數據可以導出")
            return
        
        dialog = ExportDialog(self.data, self, self.theme_manager)
        dialog.exec()
    
    def show_character_analysis(self):
        """顯示字符分析"""
        if not self.data:
            QMessageBox.information(self, "分析", "沒有數據可以分析")
            return
        
        dialog = CharacterAnalysisDialog(self.data, self, self.theme_manager)
        dialog.exec()
    
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