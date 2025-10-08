#!/bin/bash

echo "🧪 测试长宽比适配功能"
echo "================================"
echo ""
echo "📝 使用同一张签名图片，裁剪成不同长宽比："
echo "   - 1:1 正方形"
echo "   - 2:1 横向矩形"
echo "   - 1:2 纵向矩形"
echo "   - 3:1 超宽矩形"
echo "   - 1:3 超高矩形"
echo ""
echo "🎯 预期结果：算法自动标准化长宽比，匹配率>90%"
echo ""

python3 test_aspect_ratio.py
