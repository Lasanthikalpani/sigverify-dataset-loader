"""
Generate visual birth certificate images with QR codes
"""

from PIL import Image, ImageDraw, ImageFont
import qrcode
import json
import os
from datetime import datetime


class CertificateGenerator:
    """Generate birth certificate images with QR codes"""
    
    def __init__(self, output_dir='results/certificates'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_certificate(self, content, document_id, qr_data, 
                             person_name, dob, place):
        """Generate a visual birth certificate image"""
        
        # Create blank certificate (A4 size: 800x1000 pixels)
        width, height = 800, 1000
        certificate = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(certificate)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            header_font = ImageFont.truetype("arial.ttf", 24)
            text_font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Border
        draw.rectangle([(20, 20), (width-20, height-20)], outline='navy', width=3)
        draw.rectangle([(30, 30), (width-30, height-30)], outline='navy', width=1)
        
        # Header
        draw.text((width//2, 60), "DEMOCRATIC SOCIALIST", 
                  fill='navy', font=header_font, anchor='mm')
        draw.text((width//2, 90), "REPUBLIC OF SRI LANKA", 
                  fill='navy', font=header_font, anchor='mm')
        
        draw.line([(100, 120), (width-100, 120)], fill='navy', width=2)
        
        draw.text((width//2, 160), "BIRTH CERTIFICATE", 
                  fill='darkred', font=title_font, anchor='mm')
        
        draw.line([(200, 190), (width-200, 190)], fill='navy', width=1)
        
        # Certificate Details
        y_position = 230
        
        draw.text((80, y_position), "Registration No:", fill='black', font=text_font)
        draw.text((300, y_position), document_id, fill='navy', font=text_font)
        y_position += 40
        
        draw.text((80, y_position), "Name:", fill='black', font=text_font)
        draw.text((300, y_position), person_name, fill='black', font=text_font)
        y_position += 40
        
        draw.text((80, y_position), "Date of Birth:", fill='black', font=text_font)
        draw.text((300, y_position), dob, fill='black', font=text_font)
        y_position += 40
        
        draw.text((80, y_position), "Place of Birth:", fill='black', font=text_font)
        draw.text((300, y_position), place, fill='black', font=text_font)
        y_position += 60
        
        # Content Hash
        draw.text((80, y_position), "Content Hash:", fill='black', font=text_font)
        y_position += 25
        
        hash_text = qr_data['content_hash']
        for i in range(0, len(hash_text), 64):
            draw.text((100, y_position), hash_text[i:i+64], 
                      fill='gray', font=small_font)
            y_position += 20
        
        y_position += 20
        
        # QR Code
        qr = qrcode.QRCode(version=1, box_size=4, border=2)
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_image = qr_image.convert('RGB')
        qr_image = qr_image.resize((200, 200))
        
        qr_position = (width - 280, height - 320)
        certificate.paste(qr_image, qr_position)
        
        draw.text((width - 180, height - 100), "Scan to Verify", 
                  fill='navy', font=text_font, anchor='mm')
        
        # Footer
        draw.line([(50, height-60), (width-50, height-60)], fill='navy', width=1)
        
        draw.text((width//2, height-40), 
                  f"Issued by: {qr_data['issuer']}", 
                  fill='black', font=small_font, anchor='mm')
        
        draw.text((width//2, height-25), 
                  f"Timestamp: {qr_data['timestamp']}", 
                  fill='gray', font=small_font, anchor='mm')
        
        # Save
        filename = f"{self.output_dir}/{document_id}.png"
        certificate.save(filename)
        
        return filename
    
    def generate_multiple(self, certificates_data):
        """Generate multiple certificates"""
        generated_files = []
        
        for cert in certificates_data:
            filename = self.generate_certificate(
                content=cert['content'],
                document_id=cert['document_id'],
                qr_data=cert['qr_data'],
                person_name=cert['person_name'],
                dob=cert['dob'],
                place=cert['place']
            )
            generated_files.append(filename)
            print(f"✅ Generated: {filename}")
        
        return generated_files