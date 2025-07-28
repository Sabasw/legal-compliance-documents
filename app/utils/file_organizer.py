import os
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
import uuid
from pathlib import Path

class FileOrganizer:
    """Organizes generated files into structured folders"""
    
    def __init__(self, base_dir: str = "reports"):
        self.base_dir = base_dir
        self.ensure_base_dir()
    
    def ensure_base_dir(self):
        """Ensure the base directory exists"""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_document_folder(self, doc_id: str) -> str:
        """Create a folder for a specific document"""
        folder_name = f"doc_{doc_id}"
        folder_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def get_document_folder(self, doc_id: str) -> Optional[str]:
        """Get the folder path for a document"""
        folder_name = f"doc_{doc_id}"
        folder_path = os.path.join(self.base_dir, folder_name)
        if os.path.exists(folder_path):
            return folder_path
        return None
    
    def move_file_to_folder(self, file_path: str, folder_path: str, new_name: Optional[str] = None) -> Optional[str]:
        """Move a file to a specific folder"""
        try:
            if not file_path or not os.path.exists(file_path):
                return None
            
            filename = new_name or os.path.basename(file_path)
            new_path = os.path.join(folder_path, filename)
            
            # Copy file to new location
            shutil.copy2(file_path, new_path)
            
            # Optionally remove original file
            # os.remove(file_path)
            
            return new_path
        except Exception as e:
            print(f"Error moving file {file_path}: {e}")
            return None
    
    def organize_document_files(self, doc_id: str, file_paths: Dict[str, Union[str, List[str]]]) -> Dict[str, Union[str, List[str]]]:
        """Organize all files for a document into a folder"""
        try:
            folder_path = self.create_document_folder(doc_id)
            organized_files = {}
            
            print(f"Debug - Organizing files for doc {doc_id} in folder {folder_path}")
            print(f"Debug - File paths: {file_paths}")
            
            for file_type, file_path in file_paths.items():
                print(f"Debug - Processing {file_type}: {file_path} (type: {type(file_path)})")
                if file_path:
                    if isinstance(file_path, list):
                        # Handle list of files (e.g., visualization_paths)
                        organized_paths = []
                        for path in file_path:
                            print(f"Debug - Processing list item: {path}")
                            if path and os.path.exists(path):
                                new_path = self.move_file_to_folder(path, folder_path)
                                if new_path:
                                    organized_paths.append(new_path)
                        organized_files[file_type] = organized_paths
                    elif isinstance(file_path, str):
                        # Handle single file path
                        print(f"Debug - Processing string path: {file_path}")
                        if os.path.exists(file_path):
                            new_path = self.move_file_to_folder(file_path, folder_path)
                            if new_path:
                                organized_files[file_type] = new_path
            
            print(f"Debug - Organized files result: {organized_files}")
            return organized_files
        except Exception as e:
            print(f"Error organizing files for document {doc_id}: {e}")
            return {}
    

    
    def list_document_files(self, doc_id: str) -> List[str]:
        """List all files in a document's folder"""
        folder_path = self.get_document_folder(doc_id)
        if not folder_path:
            return []
        
        files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                files.append(file_path)
        
        return files
    
    def cleanup_old_files(self, days: int = 30):
        """Clean up files older than specified days"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                folder_time = os.path.getctime(folder_path)
                if folder_time < cutoff_time:
                    shutil.rmtree(folder_path)
                    print(f"Cleaned up old folder: {folder}")
    
    def manage_pdf_report(self, doc_id: str, pdf_path: str) -> str:
        """Manage PDF report - replace existing or create new"""
        folder_path = self.get_document_folder(doc_id)
        if not folder_path:
            folder_path = self.create_document_folder(doc_id)
        
        # Remove existing PDF reports
        for file in os.listdir(folder_path):
            if file.endswith('.pdf') and file.startswith('document_report_'):
                old_pdf_path = os.path.join(folder_path, file)
                try:
                    os.remove(old_pdf_path)
                    print(f"Removed old PDF report: {old_pdf_path}")
                except Exception as e:
                    print(f"Error removing old PDF: {e}")
        
        # Move new PDF to document folder
        pdf_filename = f"document_report_{doc_id}.pdf"
        new_pdf_path = os.path.join(folder_path, pdf_filename)
        
        try:
            shutil.copy2(pdf_path, new_pdf_path)
            print(f"PDF report stored: {new_pdf_path}")
            return new_pdf_path
        except Exception as e:
            print(f"Error storing PDF report: {e}")
            return pdf_path
    
    def get_pdf_report_path(self, doc_id: str) -> Optional[str]:
        """Get the path of the PDF report for a document"""
        folder_path = self.get_document_folder(doc_id)
        if not folder_path:
            return None
        
        pdf_filename = f"document_report_{doc_id}.pdf"
        pdf_path = os.path.join(folder_path, pdf_filename)
        
        if os.path.exists(pdf_path):
            return pdf_path
        return None 