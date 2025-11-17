// 全局状态
const state = {
    verificationType: 'signature',
    algorithm: 'gnn',  // 默认使用GNN算法
    templateImage: null,
    queryImage: null,
    templateCropped: null,
    queryCropped: null,
    currentTool: { template: 'rectangle', query: 'rectangle' }
};

// Canvas管理类
class CanvasManager {
    constructor(canvasId, containerId, toolsId, cropPreviewId, zoomInfoId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.containerId = containerId;
        this.toolsId = toolsId;
        this.cropPreviewId = cropPreviewId;
        this.zoomInfoId = zoomInfoId;
        this.image = null;
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentTool = 'rectangle';
        this.selection = null;
        this.scale = 1.0;
        this.minScale = 0.5;
        this.maxScale = 3.0;
        
        this.initEvents();
    }
    
    initEvents() {
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
    }
    
    loadImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    this.image = img;
                    this.scale = 1.0;
                    this.selection = null;
                    this.drawImage();
                    document.getElementById(this.containerId).style.display = 'block';
                    document.getElementById(this.toolsId).style.display = 'block';
                    this.updateZoomInfo();
                    resolve();
                };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    drawImage() {
        if (!this.image) return;
        
        // 保持原始尺寸,不自动缩放
        // 只应用用户手动的缩放操作
        const scaledWidth = this.image.width * this.scale;
        const scaledHeight = this.image.height * this.scale;
        
        this.canvas.width = scaledWidth;
        this.canvas.height = scaledHeight;
        this.ctx.clearRect(0, 0, scaledWidth, scaledHeight);
        this.ctx.drawImage(this.image, 0, 0, scaledWidth, scaledHeight);
        
        // 重绘选择框
        if (this.selection) {
            this.redrawSelection();
        }
    }
    
    redrawSelection() {
        if (!this.selection) return;
        
        this.ctx.strokeStyle = '#667eea';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([8, 8]);
        
        const startX = this.selection.startX * this.scale;
        const startY = this.selection.startY * this.scale;
        const endX = this.selection.endX * this.scale;
        const endY = this.selection.endY * this.scale;
        
        if (this.selection.tool === 'rectangle') {
            const width = endX - startX;
            const height = endY - startY;
            this.ctx.strokeRect(startX, startY, width, height);
        } else if (this.selection.tool === 'circle') {
            const radius = Math.sqrt(
                Math.pow(endX - startX, 2) + 
                Math.pow(endY - startY, 2)
            );
            this.ctx.beginPath();
            this.ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }
    
    zoom(delta) {
        const oldScale = this.scale;
        this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale + delta));
        
        if (oldScale !== this.scale) {
            this.drawImage();
            this.updateZoomInfo();
        }
    }
    
    resetZoom() {
        this.scale = 1.0;
        this.drawImage();
        this.updateZoomInfo();
    }
    
    updateZoomInfo() {
        const zoomPercent = Math.round(this.scale * 100);
        document.getElementById(this.zoomInfoId).textContent = zoomPercent + '%';
    }
    
    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.startX = (e.clientX - rect.left) / this.scale;
        this.startY = (e.clientY - rect.top) / this.scale;
        this.isDrawing = true;
    }
    
    handleMouseMove(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = (e.clientX - rect.left) / this.scale;
        const currentY = (e.clientY - rect.top) / this.scale;
        
        // 重绘图像
        this.drawImage();
        
        // 绘制当前选择框
        this.ctx.strokeStyle = '#667eea';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([8, 8]);
        
        if (this.currentTool === 'rectangle') {
            const width = (currentX - this.startX) * this.scale;
            const height = (currentY - this.startY) * this.scale;
            this.ctx.strokeRect(
                this.startX * this.scale, 
                this.startY * this.scale, 
                width, 
                height
            );
        } else if (this.currentTool === 'circle') {
            const radius = Math.sqrt(
                Math.pow((currentX - this.startX) * this.scale, 2) + 
                Math.pow((currentY - this.startY) * this.scale, 2)
            );
            this.ctx.beginPath();
            this.ctx.arc(
                this.startX * this.scale, 
                this.startY * this.scale, 
                radius, 
                0, 
                2 * Math.PI
            );
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }
    
    handleMouseUp(e) {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        
        const rect = this.canvas.getBoundingClientRect();
        const endX = (e.clientX - rect.left) / this.scale;
        const endY = (e.clientY - rect.top) / this.scale;
        
        // 保存选择区域（使用原始坐标，不含缩放）
        this.selection = {
            startX: this.startX,
            startY: this.startY,
            endX: endX,
            endY: endY,
            tool: this.currentTool
        };
    }
    
    cropImage() {
        if (!this.selection || !this.image) return null;
        
        // 重要：坐标需要除以scale，因为selection存储的是画布坐标，需要转换为原图坐标
        const scaleInverse = 1 / this.scale;
        
        if (this.selection.tool === 'rectangle') {
            const x = Math.min(this.selection.startX, this.selection.endX) * scaleInverse;
            const y = Math.min(this.selection.startY, this.selection.endY) * scaleInverse;
            const width = Math.abs(this.selection.endX - this.selection.startX) * scaleInverse;
            const height = Math.abs(this.selection.endY - this.selection.startY) * scaleInverse;
            
            // 创建临时canvas裁剪
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(this.image, x, y, width, height, 0, 0, width, height);
            
            return tempCanvas.toDataURL('image/png');
        } else if (this.selection.tool === 'circle') {
            const centerX = this.selection.startX * scaleInverse;
            const centerY = this.selection.startY * scaleInverse;
            const radius = Math.sqrt(
                Math.pow((this.selection.endX - this.selection.startX) * scaleInverse, 2) + 
                Math.pow((this.selection.endY - this.selection.startY) * scaleInverse, 2)
            );
            
            // 创建临时canvas裁剪圆形
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = radius * 2;
            tempCanvas.height = radius * 2;
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCtx.beginPath();
            tempCtx.arc(radius, radius, radius, 0, 2 * Math.PI);
            tempCtx.clip();
            
            tempCtx.drawImage(
                this.image, 
                centerX - radius, centerY - radius, radius * 2, radius * 2,
                0, 0, radius * 2, radius * 2
            );
            
            return tempCanvas.toDataURL('image/png');
        }
        
        return null;
    }
    
    reset() {
        this.selection = null;
        this.drawImage();
        // 隐藏裁剪预览
        document.getElementById(this.cropPreviewId).style.display = 'none';
    }
    
    clear() {
        this.image = null;
        this.selection = null;
        this.scale = 1.0;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        document.getElementById(this.containerId).style.display = 'none';
        document.getElementById(this.toolsId).style.display = 'none';
        document.getElementById(this.cropPreviewId).style.display = 'none';
    }
}

// 初始化Canvas管理器
const templateManager = new CanvasManager(
    'templateCanvas', 
    'templateCanvasContainer',
    'templateTools',
    'templateCropPreview',
    'templateZoomInfo'
);
const queryManager = new CanvasManager(
    'queryCanvas',
    'queryCanvasContainer', 
    'queryTools',
    'queryCropPreview',
    'queryZoomInfo'
);

// 类型选择
document.querySelectorAll('.type-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.verificationType = btn.dataset.type;
        
        // 显示/隐藏算法选择器
        const algorithmSelector = document.getElementById('algorithmSelector');
        if (state.verificationType === 'signature') {
            algorithmSelector.style.display = 'block';
        } else {
            algorithmSelector.style.display = 'none';
        }
        
        // 更新工具提示
        if (state.verificationType === 'seal') {
            // 图章推荐圆形工具
            templateManager.currentTool = 'circle';
            queryManager.currentTool = 'circle';
        } else {
            // 签名推荐矩形工具
            templateManager.currentTool = 'rectangle';
            queryManager.currentTool = 'rectangle';
        }
    });
});

// 算法选择
const algorithmSelect = document.getElementById('algorithmSelect');
const algorithmDesc = document.getElementById('algorithmDesc');

if (algorithmSelect) {
    algorithmSelect.addEventListener('change', (e) => {
        state.algorithm = e.target.value;
        
        // 更新算法描述
        const descriptions = {
            'signet': '适合专业签名验证',
            'gnn': '基于关键点结构,对旋转/缩放鲁棒',
            'clip': '通用视觉模型,适合多样化图像'
        };
        
        algorithmDesc.textContent = descriptions[state.algorithm] || '';
        console.log('算法切换到:', state.algorithm);
    });
}

// 文件上传
document.getElementById('templateInput').addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        state.templateImage = e.target.files[0];
        await templateManager.loadImage(state.templateImage);
        document.getElementById('templateReupload').style.display = 'inline-block';
        checkReadyToVerify();
    }
});

document.getElementById('queryInput').addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        state.queryImage = e.target.files[0];
        await queryManager.loadImage(state.queryImage);
        document.getElementById('queryReupload').style.display = 'inline-block';
        checkReadyToVerify();
    }
});

// 重新上传按钮
document.getElementById('templateReupload').addEventListener('click', () => {
    document.getElementById('templateInput').value = '';
    templateManager.clear();
    state.templateImage = null;
    state.templateCropped = null;
    document.getElementById('templateReupload').style.display = 'none';
    document.getElementById('templateCroppedFinal').src = '';
    checkReadyToVerify();
    updateCroppedComparison();
});

document.getElementById('queryReupload').addEventListener('click', () => {
    document.getElementById('queryInput').value = '';
    queryManager.clear();
    state.queryImage = null;
    state.queryCropped = null;
    document.getElementById('queryReupload').style.display = 'none';
    document.getElementById('queryCroppedFinal').src = '';
    checkReadyToVerify();
    updateCroppedComparison();
});

// 缩放控制
document.querySelectorAll('.zoom-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const action = btn.dataset.action;
        const target = btn.dataset.target;
        const manager = target === 'template' ? templateManager : queryManager;
        
        if (action === 'zoom-in') {
            manager.zoom(0.2);
        } else if (action === 'zoom-out') {
            manager.zoom(-0.2);
        } else if (action === 'reset') {
            manager.resetZoom();
        }
    });
});

// 工具按钮
document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tool = btn.dataset.tool;
        const parentId = btn.parentElement.parentElement.id;
        const manager = parentId === 'templateTools' ? templateManager : queryManager;
        const previewId = parentId === 'templateTools' ? 'templateCropPreview' : 'queryCropPreview';
        const previewImgId = parentId === 'templateTools' ? 'templateCropImage' : 'queryCropImage';
        const finalImgId = parentId === 'templateTools' ? 'templateCroppedFinal' : 'queryCroppedFinal';
        
        if (tool === 'crop') {
            // 执行裁剪
            const cropped = manager.cropImage();
            if (cropped) {
                // 保存裁剪结果
                if (manager === templateManager) {
                    state.templateCropped = cropped;
                } else {
                    state.queryCropped = cropped;
                }
                
                // 显示裁剪预览
                document.getElementById(previewId).style.display = 'block';
                document.getElementById(previewImgId).src = cropped;
                document.getElementById(finalImgId).src = cropped;
                
                // 更新对比区域
                updateCroppedComparison();
                
                checkReadyToVerify();
            } else {
                alert('❌ 请先选择区域！请用鼠标在图片上拖动选择要裁剪的区域');
            }
        } else if (tool === 'reset') {
            manager.reset();
        } else {
            // 切换工具
            const toolGroup = btn.parentElement;
            toolGroup.querySelectorAll('.tool-btn').forEach(b => {
                if (b.dataset.tool === 'rectangle' || b.dataset.tool === 'circle') {
                    b.classList.remove('active');
                }
            });
            btn.classList.add('active');
            manager.currentTool = tool;
        }
    });
});

// 更新裁剪对比区域
function updateCroppedComparison() {
    const comparisonDiv = document.getElementById('croppedComparison');
    if (state.templateCropped && state.queryCropped) {
        comparisonDiv.style.display = 'block';
        // 滚动到对比区域
        setTimeout(() => {
            comparisonDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    } else {
        comparisonDiv.style.display = 'none';
    }
}

// 检查是否可以验证
function checkReadyToVerify() {
    const ready = state.templateCropped && state.queryCropped;
    document.getElementById('verifyBtn').disabled = !ready;
}

// 验证按钮
document.getElementById('verifyBtn').addEventListener('click', async () => {
    // 隐藏之前的结果
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    
    try {
        console.log('🚀 开始验证...');
        console.log('验证类型:', state.verificationType);
        
        // 将base64转换为Blob
        const templateBlob = await fetch(state.templateCropped).then(r => r.blob());
        const queryBlob = await fetch(state.queryCropped).then(r => r.blob());
        
        console.log('✅ 图片转换完成');
        console.log('模板图片大小:', templateBlob.size, 'bytes');
        console.log('待验证图片大小:', queryBlob.size, 'bytes');
        
        // 创建FormData
        const formData = new FormData();
        formData.append('template_image', templateBlob, 'template.png');
        formData.append('query_image', queryBlob, 'query.png');
        formData.append('verification_type', state.verificationType);
        formData.append('algorithm', state.algorithm);  // 新增: 发送算法选择
        
        console.log('📤 发送请求到后端...');
        console.log('   - 验证类型:', state.verificationType);
        console.log('   - 算法:', state.algorithm);
        
        // 发送请求 (使用相对路径,支持单容器部署)
        const response = await fetch('/api/verify', {
            method: 'POST',
            body: formData,
            mode: 'cors',
            credentials: 'include'
        });
        
        console.log('📥 收到响应，状态码:', response.status);
        
        const result = await response.json();
        console.log('📊 验证结果:', result);
        
        document.getElementById('loading').style.display = 'none';
        
        // 检查响应状态
        if (result.success === false) {
            // 后端明确返回失败
            document.getElementById('error').textContent = '验证失败: ' + (result.error || '未知错误');
            document.getElementById('error').style.display = 'block';
        } else if (result.success === undefined && !result.final_score && !result.fast_reject) {
            // 缺少关键字段,可能是错误响应
            document.getElementById('error').textContent = '验证失败: 响应数据不完整 (请检查后端服务)';
            document.getElementById('error').style.display = 'block';
        } else {
            // 正常结果或快速拒绝
            displayResult(result);
        }
        
    } catch (error) {
        console.error('❌ 验证错误:', error);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('error').textContent = '网络错误: ' + error.message + ' (请检查服务是否正常运行)';
        document.getElementById('error').style.display = 'block';
        console.error('Verification error:', error);
    }
});

// 显示结果
function displayResult(result) {
    const resultSection = document.getElementById('resultSection');
    const scoreElement = document.getElementById('resultScore');
    
    // 🔥 检查是否快速拒绝
    if (result.fast_reject) {
        scoreElement.textContent = '0.0%';
        scoreElement.className = 'result-score score-low';
        
        const typeName = result.type === 'signature' ? '手写签名' : '印章图章';
        document.getElementById('resultType').textContent = `${typeName} · ⚡ 快速拒绝`;
        document.getElementById('algorithmType').textContent = '⚡ 笔画筛选器';
        document.getElementById('similarityScore').textContent = '0.0%';
        document.getElementById('euclideanDistance').textContent = 'N/A';
        document.getElementById('processingTime').textContent = result.processing_time_ms + 'ms';
        
        // 显示拒绝原因和特征差异
        const recommendationElement = document.getElementById('recommendation');
        const diffs = result.stroke_features.differences;
        const template = result.stroke_features.template;
        const query = result.stroke_features.query;
        
        recommendationElement.innerHTML = `
            <div style="padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;">
                <strong>⚡ 快速拒绝原因:</strong> ${result.reject_reason}
            </div>
            
            <strong>📊 笔画特征对比:</strong>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">特征</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">模板图片</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">待验证图片</th>
                        <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">差异程度</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background: ${diffs.stroke_count_diff > 0.45 ? '#fee' : '#efe'};">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>笔画数量</strong></td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${template.stroke_count}</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${query.stroke_count}</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: ${diffs.stroke_count_diff > 0.45 ? '#dc3545' : '#28a745'}">
                            ${(diffs.stroke_count_diff * 100).toFixed(1)}% ${diffs.stroke_count_diff > 0.45 ? '❌' : '✓'}
                        </td>
                    </tr>
                    <tr style="background: ${diffs.density_diff > 0.50 ? '#fee' : '#efe'};">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>签名密度</strong></td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${(template.density * 100).toFixed(2)}%</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${(query.density * 100).toFixed(2)}%</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: ${diffs.density_diff > 0.50 ? '#dc3545' : '#28a745'}">
                            ${(diffs.density_diff * 100).toFixed(1)}% ${diffs.density_diff > 0.50 ? '❌' : '✓'}
                        </td>
                    </tr>
                    <tr style="background: ${diffs.aspect_ratio_diff > 0.50 ? '#fee' : '#efe'};">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>宽高比</strong></td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${template.aspect_ratio.toFixed(2)}</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${query.aspect_ratio.toFixed(2)}</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: ${diffs.aspect_ratio_diff > 0.50 ? '#dc3545' : '#28a745'}">
                            ${(diffs.aspect_ratio_diff * 100).toFixed(1)}% ${diffs.aspect_ratio_diff > 0.50 ? '❌' : '✓'}
                        </td>
                    </tr>
                    <tr style="background: ${diffs.bbox_area_diff > 0.60 ? '#fee' : '#efe'};">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>图片尺寸</strong></td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${template.bbox_area} 像素</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">${query.bbox_area} 像素</td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: ${diffs.bbox_area_diff > 0.60 ? '#dc3545' : '#28a745'}">
                            ${(diffs.bbox_area_diff * 100).toFixed(1)}% ${diffs.bbox_area_diff > 0.60 ? '❌' : '✓'}
                        </td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td colspan="3" style="padding: 10px; border: 1px solid #ddd; text-align: right;"><strong>综合评分:</strong></td>
                        <td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: ${diffs.combined_score > 1.2 ? '#dc3545' : '#28a745'}; font-size: 1.1em;">
                            ${diffs.combined_score.toFixed(2)} ${diffs.combined_score > 1.2 ? '(超过阈值)' : ''}
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <div style="margin-top: 15px; padding: 12px; background: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 4px;">
                <strong>💡 智能预筛选说明:</strong><br>
                系统通过笔画特征快速分析发现两张图片差异明显,无需使用深度学习模型即可判定为不匹配。<br>
                <span style="color: #666;">✓ 节省计算资源 &nbsp; ✓ 加快响应速度 &nbsp; ✓ 降低服务器负载</span>
            </div>
        `;
        
        document.getElementById('cleanedComparison').style.display = 'none';
        resultSection.style.display = 'block';  // 修复: 直接设置display而不是移除hidden类
        return;
    }
    
    // 显示分数
    const scorePercent = (result.final_score * 100).toFixed(1);
    scoreElement.textContent = scorePercent + '%';
    
    // 根据置信度设置颜色
    scoreElement.className = 'result-score';
    if (result.confidence === 'high') {
        scoreElement.classList.add('score-high');
    } else if (result.confidence === 'medium') {
        scoreElement.classList.add('score-medium');
    } else {
        scoreElement.classList.add('score-low');
    }
    
    // 显示类型
    const typeName = result.type === 'signature' ? '手写签名' : '印章图章';
    document.getElementById('resultType').textContent = typeName + ' · 置信度: ' + result.confidence.toUpperCase();
    
    // 显示算法类型
    let algorithmName = result.algorithm || 'CLIP';
    let algorithmEmoji = '🎨 CLIP';
    
    if (algorithmName.includes('SigNet')) {
        algorithmEmoji = '🧠 SigNet';
    } else if (algorithmName === 'GNN') {
        algorithmEmoji = '🕸️ GNN';
    } else if (algorithmName === 'CLIP') {
        algorithmEmoji = '🎨 CLIP';
    }
    
    document.getElementById('algorithmType').textContent = algorithmEmoji;
    
    // 显示相似度
    const similarity = result.similarity || result.final_score;
    document.getElementById('similarityScore').textContent = (similarity * 100).toFixed(1) + '%';
    
    // 显示欧氏距离(SigNet/GNN)
    const euclideanDist = result.euclidean_distance || result.gnn_distance;
    if (euclideanDist !== null && euclideanDist !== undefined) {
        document.getElementById('euclideanDistance').textContent = euclideanDist.toFixed(4);
    } else {
        document.getElementById('euclideanDistance').textContent = 'N/A';
    }
    
    // 显示GNN关键点信息(如果使用GNN)
    if (algorithmName === 'GNN' && result.gnn_keypoints_template) {
        const kpInfo = `关键点: T=${result.gnn_keypoints_template}, Q=${result.gnn_keypoints_query}`;
        document.getElementById('euclideanDistance').textContent += ` (${kpInfo})`;
    }
    
    // 显示处理时间
    document.getElementById('processingTime').textContent = result.processing_time_ms + 'ms';
    
    // 显示清洁后的图片（仅签名验证且启用了清洁）
    const cleanedComparison = document.getElementById('cleanedComparison');
    if (result.type === 'signature' && result.debug_images && result.clean_enabled) {
        const templatePath = result.debug_images.template;
        const queryPath = result.debug_images.query;
        
        // 使用相对路径,支持单容器部署
        document.getElementById('templateCleanedImage').src = `/uploaded_samples/${templatePath}`;
        document.getElementById('queryCleanedImage').src = `/uploaded_samples/${queryPath}`;
        document.getElementById('cleanModeInfo').textContent = result.clean_mode || 'conservative';
        
        cleanedComparison.style.display = 'block';
    } else {
        cleanedComparison.style.display = 'none';
    }
    
    // 显示建议
    /*
    const recommendationElement = document.getElementById('recommendation');
    recommendationElement.innerHTML = `
        <strong>💡 验证建议:</strong><br>
        ${result.recommendation}<br>
        <small style="color: #666;">
            判断结果: ${result.is_authentic ? '✅ 可能为真实' : '❌ 可能为伪造'} 
            (阈值: ${(result.threshold * 100).toFixed(0)}%)
        </small>
    `;
    */
    // 显示结果区域
    resultSection.style.display = 'block';
    
    // 滚动到结果
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// 拖拽上传支持
function setupDragDrop(boxElement, inputElement, manager, reuploadBtn) {
    boxElement.addEventListener('dragover', (e) => {
        e.preventDefault();
        boxElement.style.borderColor = '#667eea';
        boxElement.style.background = '#f0f3ff';
    });
    
    boxElement.addEventListener('dragleave', () => {
        boxElement.style.borderColor = '#ddd';
        boxElement.style.background = '#f8f9ff';
    });
    
    boxElement.addEventListener('drop', async (e) => {
        e.preventDefault();
        boxElement.style.borderColor = '#ddd';
        boxElement.style.background = '#f8f9ff';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (manager === templateManager) {
                state.templateImage = file;
            } else {
                state.queryImage = file;
            }
            await manager.loadImage(file);
            reuploadBtn.style.display = 'inline-block';
            checkReadyToVerify();
        }
    });
}

// 初始化拖拽
setupDragDrop(
    document.getElementById('templateBox'), 
    document.getElementById('templateInput'), 
    templateManager,
    document.getElementById('templateReupload')
);
setupDragDrop(
    document.getElementById('queryBox'), 
    document.getElementById('queryInput'), 
    queryManager,
    document.getElementById('queryReupload')
);

console.log('✅ 签名图章验证系统已加载 - 增强版');
console.log('📖 使用说明:');
console.log('1. 选择验证类型(签名/图章)');
console.log('2. 上传模板和待验证图片');
console.log('3. 使用缩放按钮(+/-/⟲)调整图片大小');
console.log('4. 使用工具选择区域并裁剪');
console.log('5. 查看裁剪预览和对比');
console.log('6. 点击开始验证');
console.log('');
console.log('💡 提示:');
console.log('   - 可以随时点击"重新上传"更换图片');
console.log('   - 放大图片可以更精确地选择裁剪区域');
console.log('   - 裁剪后会显示预览，确认无误再验证');
