#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern PDF Processor

Replaces PyMuPDF (fitz) with modern alternatives:
- pdfplumber for text and table extraction
- pypdfium2 for image extraction and advanced features
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import io
import re

logger = logging.getLogger(__name__)

class ModernPDFProcessor:
    """Modern PDF processor using pdfplumber and pypdfium2 instead of PyMuPDF"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which PDF processing methods are available"""
        methods = {}
        
        # Check for pdfplumber
        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            methods['pdfplumber'] = False
        
        # Check for pypdfium2
        try:
            import pypdfium2
            methods['pypdfium2'] = True
        except ImportError:
            methods['pypdfium2'] = False
        
        # Check for pypdf
        try:
            import pypdf
            methods['pypdf'] = True
        except ImportError:
            methods['pypdf'] = False
        
        return methods
    
    def process_pdf(self, file_path: str = None, file_content: bytes = None) -> Dict[str, Any]:
        """Process PDF using modern libraries"""
        
        if not file_path and not file_content:
            raise ValueError("Either file_path or file_content must be provided")
        
        # Check if modern libraries are available
        if not self.available_methods.get('pdfplumber'):
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': "pdfplumber not available",
                'suggestion': "Install with: pip install pdfplumber"
            }
        
        try:
            # Extract text and tables using pdfplumber
            text_content, page_count, tables = self._extract_text_and_tables(file_path, file_content)
            
            # Extract images using pypdfium2 if available
            images = []
            if self.available_methods.get('pypdfium2'):
                images = self._extract_images(file_path, file_content)
            
            return {
                'success': True,
                'confidence_score': 0.90,
                'processing': {
                    'text_content': text_content,
                    'page_count': page_count,
                    'images': images,
                    'tables': tables,
                    'file_size': len(file_content) if file_content else Path(file_path).stat().st_size,
                    'method': 'modern_pdf_processor'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def _extract_text_and_tables(self, file_path: str = None, file_content: bytes = None) -> tuple:
        """Extract text and tables using pdfplumber"""
        import pdfplumber
        
        if file_content:
            pdf = pdfplumber.open(io.BytesIO(file_content))
        else:
            pdf = pdfplumber.open(file_path)
        
        text_content = ""
        page_count = len(pdf.pages)
        tables = []
        
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            page_text = page.extract_text() or ""
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Extract tables
            page_tables = page.extract_tables()
            for table_index, table in enumerate(page_tables):
                table_data = {
                    "page": page_num + 1,
                    "index": table_index,
                    "rows": len(table),
                    "columns": len(table[0]) if table else 0,
                    "content": table,
                    "header": table[0] if table else [],
                    "type": self._classify_table_type(table)
                }
                tables.append(table_data)
        
        pdf.close()
        return text_content, page_count, tables
    
    def _extract_images(self, file_path: str = None, file_content: bytes = None) -> List[Dict[str, Any]]:
        """Extract images using pypdfium2 with fallback to specialized worker"""
        try:
            import pypdfium2
            
            if file_content:
                pdf = pypdfium2.PdfDocument(file_content)
            else:
                pdf = pypdfium2.PdfDocument(file_path)
            
            images = []
            
            try:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    
                    # Get image list for this page
                    try:
                        # Try the correct API method
                        page_images = page.get_images()
                        
                        for img_index, img_info in enumerate(page_images):
                            try:
                                # Get image data
                                img_data = page.get_image(img_index)
                                
                                if img_data:
                                    img_metadata = {
                                        "page": page_num + 1,
                                        "index": img_index,
                                        "width": getattr(img_info, 'width', 0),
                                        "height": getattr(img_info, 'height', 0),
                                        "size_bytes": len(img_data),
                                        "type": self._classify_image_type(
                                            getattr(img_info, 'width', 0), 
                                            getattr(img_info, 'height', 0)
                                        ),
                                        "format": getattr(img_info, 'format', 'unknown')
                                    }
                                    images.append(img_metadata)
                            
                            except Exception as e:
                                logger.warning(f"Error processing image on page {page_num + 1}: {e}")
                                
                    except AttributeError:
                        # Fallback: try to get images using different method
                        try:
                            # Some versions of pypdfium2 have different API
                            page_images = page.get_image_info()
                            
                            for img_index, img_info in enumerate(page_images):
                                try:
                                    img_data = page.get_image(img_info.index)
                                    
                                    if img_data:
                                        img_metadata = {
                                            "page": page_num + 1,
                                            "index": img_index,
                                            "width": img_info.width,
                                            "height": img_info.height,
                                            "size_bytes": len(img_data),
                                            "type": self._classify_image_type(img_info.width, img_info.height),
                                            "format": img_info.format
                                        }
                                        images.append(img_metadata)
                                
                                except Exception as e:
                                    logger.warning(f"Error processing image on page {page_num + 1}: {e}")
                                    
                        except Exception as e:
                            logger.warning(f"Could not extract images from page {page_num + 1}: {e}")
                
                # If we got images, return them
                if images:
                    return images
                
                # If no images found with pypdfium2, try specialized worker
                logger.info("No images found with pypdfium2, trying specialized image worker...")
                return self._try_specialized_image_worker(file_path, file_content)
            
            finally:
                # Ensure PDF is properly closed
                pdf.close()
            
        except Exception as e:
            logger.error(f"Error extracting images with pypdfium2: {e}")
            # Try specialized worker as fallback
            logger.info("pypdfium2 failed, trying specialized image worker...")
            return self._try_specialized_image_worker(file_path, file_content)
    
    def _try_specialized_image_worker(self, file_path: str = None, file_content: bytes = None) -> List[Dict[str, Any]]:
        """Try specialized image processing worker when primary method fails"""
        try:
            from .image_processing_worker import ImageProcessingWorker
            
            worker = ImageProcessingWorker()
            result = worker.process_images(file_path, file_content)
            
            if result['success']:
                logger.info(f"Specialized worker extracted {result['total_images']} images using {result['processing_method']}")
                return result['images']
            else:
                logger.warning(f"Specialized worker failed: {result.get('methods_tried', [])}")
                return []
                
        except ImportError as e:
            logger.warning(f"Image processing worker not available: {e}")
            return []
        except Exception as e:
            logger.error(f"Error with specialized image worker: {e}")
            return []
    
    def _classify_image_type(self, width: int, height: int) -> str:
        """Classify image type based on dimensions"""
        aspect_ratio = width / height if height > 0 else 0
        
        if aspect_ratio > 1.5:
            return "chart"
        elif width < 200 and height < 200:
            return "icon"
        elif width > 800 and height > 600:
            return "diagram"
        else:
            return "image"
    
    def _classify_table_type(self, table_content: List[List[str]]) -> str:
        """Classify table type"""
        if not table_content:
            return "unknown"
        
        numeric_count = 0
        total_cells = 0
        
        for row in table_content:
            for cell in row:
                if cell:  # Check if cell is not empty
                    total_cells += 1
                    if re.search(r'\d+\.?\d*', str(cell)):
                        numeric_count += 1
        
        numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0
        
        if numeric_ratio > 0.5:
            return "data_table"
        elif len(table_content) > 10:
            return "reference_table"
        else:
            return "summary_table"
