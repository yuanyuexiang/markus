#!/bin/bash
# 测试签名清洁功能

echo "🧪 签名清洁功能测试"
echo "===================="
echo ""

# 测试文件路径（请根据实际情况修改）
TEMPLATE="uploaded_samples/signature_template_20250129_003621.png"
QUERY="uploaded_samples/signature_query_20250129_003621.png"

if [ ! -f "$TEMPLATE" ] || [ ! -f "$QUERY" ]; then
    echo "⚠️  测试文件不存在，请先通过前端上传两张签名图片"
    echo "   模板: $TEMPLATE"
    echo "   查询: $QUERY"
    echo ""
    echo "💡 使用方法："
    echo "   1. 在浏览器打开 http://localhost:3000"
    echo "   2. 上传两张裁剪的签名图片"
    echo "   3. 查看 uploaded_samples/ 目录中生成的文件"
    echo "   4. 修改此脚本中的文件路径并重新运行"
    exit 1
fi

echo "📁 使用测试文件:"
echo "   模板: $TEMPLATE"
echo "   查询: $QUERY"
echo ""

# 测试1: 不启用清洁
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试1: 不启用清洁 (enable_clean=false)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "enable_clean=false" | jq '{
    algorithm: .algorithm,
    similarity: .similarity,
    euclidean_distance: .euclidean_distance,
    ssim: .ssim,
    pipeline: .signet_pipeline,
    clean_enabled: .clean_enabled,
    is_authentic: .is_authentic
  }'
echo ""

# 测试2: 启用保守清洁（中文签名）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试2: 启用保守清洁 (conservative - 中文)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESULT=$(curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "enable_clean=true" \
  -F "clean_mode=conservative")

echo "$RESULT" | jq '{
    algorithm: .algorithm,
    similarity: .similarity,
    euclidean_distance: .euclidean_distance,
    ssim: .ssim,
    pipeline: .signet_pipeline,
    clean_enabled: .clean_enabled,
    clean_mode: .clean_mode,
    is_authentic: .is_authentic,
    debug_images: .debug_images
  }'

# 显示调试图像路径
DEBUG_TEMPLATE=$(echo "$RESULT" | jq -r '.debug_images.template // empty')
DEBUG_QUERY=$(echo "$RESULT" | jq -r '.debug_images.query // empty')

if [ -n "$DEBUG_TEMPLATE" ]; then
    echo ""
    echo "🖼️  清洁后的图像已保存:"
    echo "   模板: uploaded_samples/$DEBUG_TEMPLATE"
    echo "   查询: uploaded_samples/$DEBUG_QUERY"
fi

echo ""

# 测试3: 启用激进清洁（英文签名）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试3: 启用激进清洁 (aggressive - 英文)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST http://localhost:8000/api/verify \
  -F "template_image=@$TEMPLATE" \
  -F "query_image=@$QUERY" \
  -F "verification_type=signature" \
  -F "enable_clean=true" \
  -F "clean_mode=aggressive" | jq '{
    algorithm: .algorithm,
    similarity: .similarity,
    euclidean_distance: .euclidean_distance,
    ssim: .ssim,
    pipeline: .signet_pipeline,
    clean_enabled: .clean_enabled,
    clean_mode: .clean_mode,
    is_authentic: .is_authentic
  }'

echo ""
echo "✅ 测试完成！"
echo ""
echo "💡 提示:"
echo "   - 对比三种模式的相似度和欧氏距离"
echo "   - 查看 uploaded_samples/debug/ 目录中的清洁后图像"
echo "   - 如果清洁后相似度提高，说明成功去除了杂质"
