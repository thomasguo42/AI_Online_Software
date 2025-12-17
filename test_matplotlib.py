import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

def diagnose_chinese_fonts():
    """Diagnose available Chinese fonts and test display"""
    
    print("=== FONT DIAGNOSTIC REPORT ===")
    print(f"Operating System: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    
    # 1. List all available fonts
    print("\n1. Checking all available fonts...")
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"Total fonts available: {len(all_fonts)}")
    
    # 2. Look for Chinese-related fonts
    print("\n2. Looking for Chinese fonts...")
    chinese_keywords = ['Chinese', 'CJK', 'Han', 'Hei', 'Song', 'Ming', 'YaHei', 'SimHei', 'SimSun', 'WenQuanYi', 'PingFang', 'Hiragino', 'Noto']
    chinese_fonts = []
    
    for font in fm.fontManager.ttflist:
        for keyword in chinese_keywords:
            if keyword.lower() in font.name.lower():
                chinese_fonts.append(font.name)
                break
    
    chinese_fonts = list(set(chinese_fonts))  # Remove duplicates
    print(f"Found Chinese fonts: {chinese_fonts}")
    
    # 3. Check common font paths
    print("\n3. Checking common font file paths...")
    common_paths = [
        # Linux
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        # macOS
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        # Windows
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ]
    
    available_paths = []
    for path in common_paths:
        if os.path.exists(path):
            available_paths.append(path)
            print(f"✓ Found: {path}")
    
    if not available_paths:
        print("✗ No common Chinese font files found")
    
    # 4. Test matplotlib configuration
    print("\n4. Testing matplotlib configuration...")
    
    # Current configuration
    print(f"Current font.family: {plt.rcParams['font.family']}")
    print(f"Current font.sans-serif: {plt.rcParams['font.sans-serif']}")
    
    return chinese_fonts, available_paths

def test_chinese_display_methods():
    """Test different methods to display Chinese characters"""
    
    print("\n=== TESTING CHINESE DISPLAY METHODS ===")
    
    test_text = "击剑手表现对比"
    methods_results = {}
    
    # Method 1: Default matplotlib
    try:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, test_text, fontsize=20, ha='center', va='center')
        plt.title("Method 1: Default")
        plt.savefig('test_method1.png', dpi=150, bbox_inches='tight')
        plt.close()
        methods_results['Method 1 (Default)'] = 'test_method1.png'
        print("✓ Method 1 completed")
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
    
    # Method 2: With font.sans-serif setting
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, test_text, fontsize=20, ha='center', va='center')
        plt.title("Method 2: font.sans-serif")
        plt.savefig('test_method2.png', dpi=150, bbox_inches='tight')
        plt.close()
        methods_results['Method 2 (sans-serif)'] = 'test_method2.png'
        print("✓ Method 2 completed")
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    # Method 3: Using specific font file (if available)
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    
    for i, font_path in enumerate(font_paths, 3):
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, test_text, fontsize=20, ha='center', va='center', fontproperties=font_prop)
                plt.title(f"Method {i}: {os.path.basename(font_path)}")
                plt.savefig(f'test_method{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
                methods_results[f'Method {i} ({os.path.basename(font_path)})'] = f'test_method{i}.png'
                print(f"✓ Method {i} completed")
                break
            except Exception as e:
                print(f"✗ Method {i} failed: {e}")
    
    # Method 4: Force UTF-8 encoding
    try:
        plt.figure(figsize=(8, 6))
        # Ensure text is properly encoded
        encoded_text = test_text.encode('utf-8').decode('utf-8')
        plt.text(0.5, 0.5, encoded_text, fontsize=20, ha='center', va='center')
        plt.title("Method 4: UTF-8 encoding")
        plt.savefig('test_method4.png', dpi=150, bbox_inches='tight')
        plt.close()
        methods_results['Method 4 (UTF-8)'] = 'test_method4.png'
        print("✓ Method 4 completed")
    except Exception as e:
        print(f"✗ Method 4 failed: {e}")
    
    return methods_results

def create_robust_chinese_chart_function():
    """Create the most robust version based on system capabilities"""
    
    chinese_fonts, available_paths = diagnose_chinese_fonts()
    test_results = test_chinese_display_methods()
    
    print(f"\n=== RECOMMENDATIONS ===")
    
    if available_paths:
        print(f"✓ Use font file: {available_paths[0]}")
        recommended_method = "font_file"
        recommended_path = available_paths[0]
    elif chinese_fonts:
        print(f"✓ Use system font: {chinese_fonts[0]}")
        recommended_method = "system_font"
        recommended_font = chinese_fonts[0]
    else:
        print("⚠ No Chinese fonts detected. Install Chinese fonts:")
        print("  Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei")
        print("  CentOS/RHEL: sudo yum install wqy-microhei-fonts")
        print("  macOS: Install from App Store or manually")
        print("  Windows: Should have built-in Chinese fonts")
        recommended_method = "fallback"
    
    return recommended_method, locals()

# Run the diagnostic
if __name__ == "__main__":
    result = create_robust_chinese_chart_function()
    print("\nDiagnostic complete. Check the generated test images to see which method works.")