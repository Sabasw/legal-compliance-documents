import os
from typing import List, Dict, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import json

class PDFReportGenerator:
    """Generates PDF reports from document analysis files"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        self.code_style = ParagraphStyle(
            'CustomCode',
            parent=self.styles['Code'],
            fontSize=9,
            spaceAfter=6,
            fontName='Courier'
        )
    
    def create_document_report(self, doc_id: str, document_data: Dict, file_paths: List[str], output_path: str) -> str:
        """Create a comprehensive PDF report for a document"""
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("Document Analysis Report", self.title_style))
        story.append(Spacer(1, 20))
        
        # Document information
        story.append(Paragraph("Document Information", self.heading_style))
        story.append(Paragraph(f"Document ID: {doc_id}", self.body_style))
        story.append(Paragraph(f"Document Type: {document_data.get('document_type', 'Unknown')}", self.body_style))
        story.append(Paragraph(f"Compliance Status: {document_data.get('compliance_status', 'Unknown')}", self.body_style))
        story.append(Paragraph(f"Risk Score: {document_data.get('risk_score', 'Unknown')}", self.body_style))
        story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.body_style))
        story.append(Spacer(1, 20))
        
        # Summary
        if document_data.get('summary'):
            story.append(Paragraph("Executive Summary", self.heading_style))
            story.append(Paragraph(document_data['summary'], self.body_style))
            story.append(Spacer(1, 20))
        
        # Risk Profile
        if document_data.get('risk_profile'):
            story.append(Paragraph("Risk Profile", self.heading_style))
            risk_profile = document_data['risk_profile']
            if isinstance(risk_profile, dict):
                risk_data = []
                for risk_type, score in risk_profile.items():
                    risk_data.append([risk_type.replace('_', ' ').title(), f"{score:.2f}"])
                
                risk_table = Table(risk_data, colWidths=[2*inch, 1*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(risk_table)
                story.append(Spacer(1, 20))
        
        # Key Issues
        if document_data.get('key_issues'):
            story.append(Paragraph("Key Issues Identified", self.heading_style))
            key_issues = document_data['key_issues']
            if isinstance(key_issues, list):
                for i, issue in enumerate(key_issues[:10], 1):  # Limit to first 10 issues
                    story.append(Paragraph(f"{i}. {issue}", self.body_style))
            story.append(Spacer(1, 20))
        
        # Recommendations
        if document_data.get('recommendations'):
            story.append(Paragraph("Recommendations", self.heading_style))
            recommendations = document_data['recommendations']
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations[:10], 1):  # Limit to first 10 recommendations
                    story.append(Paragraph(f"{i}. {rec}", self.body_style))
            story.append(Spacer(1, 20))
        
        # Add images if available
        for file_path in file_paths:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Check if image exists and is not too large
                    if os.path.exists(file_path) and os.path.getsize(file_path) < 10*1024*1024:  # 10MB limit
                        img = Image(file_path, width=6*inch, height=4*inch)
                        story.append(Paragraph(f"Visualization: {os.path.basename(file_path)}", self.heading_style))
                        story.append(img)
                        story.append(Spacer(1, 20))
                except Exception as e:
                    print(f"Error adding image {file_path}: {e}")
        
        # Full Analysis (if not too long)
        if document_data.get('full_analysis'):
            story.append(Paragraph("Detailed Analysis", self.heading_style))
            full_analysis = document_data['full_analysis']
            if isinstance(full_analysis, str) and len(full_analysis) < 5000:  # Limit length
                story.append(Paragraph(full_analysis, self.body_style))
            else:
                story.append(Paragraph("Detailed analysis available in separate files.", self.body_style))
            story.append(Spacer(1, 20))
        
        # Appendices
        story.append(Paragraph("Appendices", self.heading_style))
        story.append(Paragraph("Generated Files:", self.body_style))
        for file_path in file_paths:
            if os.path.exists(file_path):
                story.append(Paragraph(f"• {os.path.basename(file_path)}", self.body_style))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def create_simple_report(self, doc_id: str, file_paths: List[str], output_path: str) -> str:
        """Create a simple PDF report listing all files"""
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Document Analysis Files", self.title_style))
        story.append(Spacer(1, 20))
        
        # File list
        story.append(Paragraph("Generated Files:", self.heading_style))
        for file_path in file_paths:
            if os.path.exists(file_path):
                story.append(Paragraph(f"• {os.path.basename(file_path)}", self.body_style))
        
        # Build PDF
        doc.build(story)
        return output_path 