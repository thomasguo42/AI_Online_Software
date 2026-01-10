#!/usr/bin/env python3
"""
Startup script for the fencing analysis Flask application with Gemini API configured
"""

import os
import sys

def setup_environment():
    """Set up environment variables"""
    # Set Gemini API key
    os.environ['GEMINI_API_KEY'] = 'AIzaSyCAKZxJCnt7BKfsBH1ImvunKuaui-2L_9U'
    
    print("🔑 Environment variables configured:")
    print(f"   GEMINI_API_KEY: {os.environ['GEMINI_API_KEY'][:10]}...")

def main():
    print("🎯 Starting Fencing Analysis Application")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Import and run the Flask app
    try:
        print("🚀 Loading Flask application...")
        import app
        print("✅ Flask application loaded successfully!")
        print("📊 Available features:")
        print("   • Video View Analysis with AI Loss Analysis")
        print("   • Touch Category Analysis")
        print("   • Performance Metrics Dashboard")
        print("   • Mirror Bar Charts")
        print("   • Gemini AI-Powered Loss Insights")
        print()
        print("🌐 Access the application at: http://localhost:5000")
        print("🎯 Navigate to any upload and click '性能分析' to use the enhanced video view")
        print()
        
        # Run the Flask app
        if __name__ == '__main__':
            app.app.run(debug=True, host='0.0.0.0', port=5000)
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()