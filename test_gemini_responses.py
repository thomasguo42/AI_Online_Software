#!/usr/bin/env python3
"""
Test script to debug Gemini API JSON parsing issues
"""

import os
import sys
sys.path.append('/workspace/Project')

from your_scripts.video_view_analysis import call_gemini_api

# Set API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyCAKZxJCnt7BKfsBH1ImvunKuaui-2L_9U'

def test_simple_prompt():
    """Test with a very simple prompt to see if it works"""
    
    simple_prompt = '''请用中文分析击剑表现并以JSON格式回复：

数据：胜率50%，共10场比赛

请返回JSON：
{
  "performance_summary": "用中文总结表现",
  "recommendations": ["中文建议1", "中文建议2"],
  "overall_rating": "评分1-10"
}'''
    
    print("🧪 Testing simple prompt...")
    print(f"Prompt: {simple_prompt}")
    print("-" * 50)
    
    result = call_gemini_api(simple_prompt, {"test": "data"})
    
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    return result

def test_complex_prompt():
    """Test with a more complex prompt similar to what we use"""
    
    complex_prompt = '''You are an expert fencing coach. Analyze this performance data and respond in Chinese with JSON format.

Performance Data:
- Win Rate: 60% (6/10)
- Category: Attack
- Wins: 6, Losses: 4

Provide analysis in this exact JSON format:
{
  "performance_summary": "用中文2-3句概述整体表现",
  "technical_analysis": "中文技术分析",
  "tactical_analysis": "中文战术分析", 
  "recommendations": [
    "中文可执行建议1",
    "中文可执行建议2",
    "中文可执行建议3"
  ],
  "training_focus": "中文训练重点",
  "overall_rating": "1-10的整数评分"
}

请使用中文作答，分点表达，简洁有力。'''
    
    print("\n🧪 Testing complex prompt...")
    print(f"Prompt length: {len(complex_prompt)} characters")
    print("-" * 50)
    
    result = call_gemini_api(complex_prompt, {"test": "complex_data"})
    
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    return result

if __name__ == "__main__":
    print("🎯 Gemini API JSON Parsing Debug Test")
    print("=" * 60)
    
    # Test 1: Simple prompt
    simple_result = test_simple_prompt()
    
    # Test 2: Complex prompt
    complex_result = test_complex_prompt()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Simple prompt success: {'✅' if isinstance(simple_result, dict) and 'performance_summary' in simple_result else '❌'}")
    print(f"Complex prompt success: {'✅' if isinstance(complex_result, dict) and 'performance_summary' in complex_result else '❌'}")