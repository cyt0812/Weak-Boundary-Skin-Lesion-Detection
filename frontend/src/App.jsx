
import { useRef, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/api/predict'
const MAX_FILE_SIZE = 10 * 1024 * 1024

const navigation = [
  { id: 'overview', label: '工作台', number: '01', icon: '⌂' },
  { id: 'analysis', label: '病灶分析', number: '02', icon: '◎' },
  { id: 'classification', label: '分类结果', number: '03', icon: '◈' },
  { id: 'evaluation', label: '模型评估', number: '04', icon: '▥' },
]

function App() {
  const [activePage, setActivePage] = useState('overview')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [message, setMessage] = useState('')
  const [dragging, setDragging] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  const fileInputRef = useRef(null)

  const currentPage =
    navigation.find((item) => item.id === activePage) ?? navigation[0]

  function handleFile(file) {
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setMessage('请选择 JPG、PNG 等图片文件。')
      return
    }

    if (file.size > MAX_FILE_SIZE) {
      setMessage('图片大小不能超过 10 MB，请选择较小的图片。')
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setAnalysisResult(null)
    setMessage('图片已载入，可以调用分类模型进行真实推理。')
    setActivePage('analysis')
  }

  function handleFileChange(event) {
    handleFile(event.target.files?.[0])
    event.target.value = ''
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setMessage('请先上传一张皮肤镜图片。')
      return
    }

    if (analyzing) return

    setAnalyzing(true)
    setAnalysisResult(null)
    setMessage('正在将图片发送给分类模型，请稍候……')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      let data

      try {
        data = await response.json()
      } catch {
        throw new Error('后端返回的内容无法解析，请检查 API 服务。')
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : `请求失败，HTTP 状态码：${response.status}`,
        )
      }

      if (
        !data.predicted_class ||
        !Array.isArray(data.scores) ||
        data.scores.length !== 7
      ) {
        throw new Error('后端返回的数据格式不符合预期。')
      }

      setAnalysisResult(data)
      setMessage('分类推理完成。以下结果来自已加载的真实模型。')
    } catch (error) {
      console.error('Classification request failed:', error)

      setAnalysisResult(null)

      if (error instanceof TypeError) {
        setMessage(
          '无法连接 Python 后端。请确认 8000 端口的服务仍在运行，然后重试。',
        )
      } else {
        setMessage(`分析失败：${error.message}`)
      }
    } finally {
      setAnalyzing(false)
    }
  }

  function handleClear() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setPreviewUrl('')
    setSelectedFile(null)
    setAnalysisResult(null)
    setMessage('')
    setDragging(false)

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  function openAnalysis() {
    setActivePage('analysis')
    setMessage('')
  }

  function navigateTo(page) {
    setActivePage(page)
    setMessage('')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#overview"
          onClick={(event) => {
            event.preventDefault()
            navigateTo('overview')
          }}
        >
          <span className="brand-mark">
            <svg viewBox="0 0 40 40" aria-hidden="true">
              <path
                d="M8 22C10 12 17 7 24 9c7 2 10 8 7 14-3 7-10 10-17 7-5-2-8-5-6-8Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path
                d="M12 24c5-5 10-6 17-5"
                fill="none"
                stroke="#a6cbbd"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </span>

          <span>
            <span className="brand-title">DermaBoundary</span>
            <span className="brand-subtitle">AI RESEARCH WORKSPACE</span>
          </span>
        </a>

        <div className="nav-caption">WORKSPACE</div>

        <nav className="side-nav" aria-label="主导航">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${
                activePage === item.id ? 'active' : ''
              }`}
              onClick={() => navigateTo(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              <span className="nav-number">{item.number}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-project">
          <div className="project-status">
            <span className="status-dot" />
            CLASSIFIER CONNECTED
          </div>

          <p>Weak Boundary Guided Skin Lesion Detection System</p>

          <span className="project-version">
            Research prototype · v0.2
          </span>
        </div>

        <div className="sidebar-footer">
          <span className="footer-dot" />
          Local development environment
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{currentPage.label}</strong>
          </div>

          <div className="topbar-right">
            <span className="environment-pill">
              <span className="status-dot" />
              LOCAL MODEL
            </span>
            <div className="avatar">DB</div>
          </div>
        </header>

        <div className="page-content">
          {activePage === 'overview' && (
            <Overview
              onStart={openAnalysis}
              onNavigate={navigateTo}
              modelConnected={Boolean(analysisResult)}
            />
          )}

          {activePage === 'analysis' && (
            <Analysis
              file={selectedFile}
              previewUrl={previewUrl}
              fileInputRef={fileInputRef}
              dragging={dragging}
              setDragging={setDragging}
              onFile={handleFile}
              onFileChange={handleFileChange}
              onAnalyze={handleAnalyze}
              analyzing={analyzing}
              analysisResult={analysisResult}
              message={message}
              onClear={handleClear}
            />
          )}

          {activePage === 'classification' && (
            <Classification
              analysisResult={analysisResult}
              onStart={openAnalysis}
            />
          )}

          {activePage === 'evaluation' && <Evaluation />}
        </div>

        <footer className="page-footer">
          <span>DERMABOUNDARY AI</span>
          <span>
            Academic research prototype · Not for clinical diagnosis
          </span>
        </footer>
      </main>
    </div>
  )
}

function PageHeading({ eyebrow, title, description, action }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action}
    </div>
  )
}

function Overview({ onStart, onNavigate }) {
  return (
    <>
      <PageHeading
        eyebrow="RESEARCH WORKSPACE / 01"
        title="病灶分析工作台"
        description="探索皮肤病灶分割、边界分析与分类方法。"
        action={
          <button className="button button-primary" onClick={onStart}>
            <span>＋</span> 新建分析
          </button>
        }
      />

      <section className="welcome-panel">
        <div className="welcome-copy">
          <div className="welcome-tag">
            <span className="welcome-tag-dot" />
            SKIN LESION RESEARCH
          </div>

          <h2>
            从图像出发，
            <br />
            <span>理解病灶边界。</span>
          </h2>

          <p>
            一个用于皮肤病灶图像分析的研究原型，逐步整合分割、分类与模型评估流程。
          </p>

          <button className="button button-light" onClick={onStart}>
            开始图像分析 <span>↗</span>
          </button>
        </div>

        <div className="welcome-graphic" aria-hidden="true">
          <div className="graphic-orbit orbit-one" />
          <div className="graphic-orbit orbit-two" />

          <div className="graphic-core">
            <svg viewBox="0 0 220 220">
              <defs>
                <linearGradient
                  id="lesionFill"
                  x1="0"
                  y1="0"
                  x2="1"
                  y2="1"
                >
                  <stop offset="0%" stopColor="#d9b6a1" />
                  <stop offset="100%" stopColor="#8c6658" />
                </linearGradient>
              </defs>

              <circle
                cx="110"
                cy="110"
                r="98"
                fill="#d8e6de"
                opacity=".28"
              />

              <path
                d="M52 105C44 80 65 55 91 59c16-19 45-8 53 9 25 7 31 34 14 52 4 24-21 43-43 34-23 14-49 0-48-20-18-5-24-18-15-29Z"
                fill="url(#lesionFill)"
                stroke="#f3e5d9"
                strokeWidth="3"
              />

              <path
                d="M52 105C44 80 65 55 91 59c16-19 45-8 53 9 25 7 31 34 14 52 4 24-21 43-43 34-23 14-49 0-48-20-18-5-24-18-15-29Z"
                fill="none"
                stroke="#e7cbb7"
                strokeWidth="2"
                strokeDasharray="4 5"
              />

              <circle cx="72" cy="91" r="5" fill="#6c4c40" opacity=".7" />
              <circle cx="121" cy="82" r="8" fill="#6c4c40" opacity=".5" />
              <circle cx="134" cy="123" r="6" fill="#51382f" opacity=".55" />
              <circle cx="88" cy="139" r="4" fill="#51382f" opacity=".5" />

              <path
                d="M157 68l22-19"
                stroke="#b9d9cb"
                strokeWidth="1.5"
              />
              <circle cx="182" cy="46" r="3" fill="#b9d9cb" />

              <path
                d="M60 154l-20 17"
                stroke="#b9d9cb"
                strokeWidth="1.5"
              />
              <circle cx="37" cy="174" r="3" fill="#b9d9cb" />
            </svg>
          </div>

          <div className="graphic-label label-top">IMAGE ANALYSIS</div>
          <div className="graphic-label label-bottom">BOUNDARY STUDY</div>
        </div>
      </section>

      <div className="section-heading">
        <div>
          <h2>研究模块</h2>
          <p>围绕毕设研究目标逐步搭建功能</p>
        </div>
        <span className="section-note">04 MODULES</span>
      </div>

      <section className="module-grid">
        <ModuleCard
          number="01"
          icon="◎"
          title="病灶分割"
          description="展示输入图像、预测掩码与边界叠加结果。"
          status="待接入模型"
          onClick={() => onNavigate('analysis')}
        />

        <ModuleCard
          number="02"
          icon="◈"
          title="特征分类"
          description="使用已接入的七分类模型输出真实预测分数。"
          status="分类模型已接入"
          onClick={() => onNavigate('classification')}
        />

        <ModuleCard
          number="03"
          icon="⌘"
          title="分割引导融合"
          description="为基础分类与融合分类的对比预留接口。"
          status="开发规划"
          onClick={() => onNavigate('classification')}
        />

        <ModuleCard
          number="04"
          icon="▥"
          title="实验评估"
          description="展示真实实验中的 Dice、IoU 与分类指标。"
          status="待接入评估数据"
          onClick={() => onNavigate('evaluation')}
        />
      </section>

      <section className="notice-panel">
        <div className="notice-icon">i</div>
        <div>
          <strong>当前系统进度</strong>
          <p>
            七分类模型已通过 Python 后端接口接入。病灶分割、边界叠加和分割引导融合尚未接入真实模型。
          </p>
        </div>
      </section>
    </>
  )
}

function ModuleCard({
  number,
  icon,
  title,
  description,
  status,
  onClick,
}) {
  return (
    <button className="module-card" onClick={onClick}>
      <div className="module-card-top">
        <span className="module-icon">{icon}</span>
        <span className="module-number">{number}</span>
      </div>

      <h3>{title}</h3>
      <p>{description}</p>

      <div className="module-card-bottom">
        <span>{status}</span>
        <span className="module-arrow">↗</span>
      </div>
    </button>
  )
}

function Analysis({
  file,
  previewUrl,
  fileInputRef,
  dragging,
  setDragging,
  onFile,
  onFileChange,
  onAnalyze,
  analyzing,
  analysisResult,
  message,
  onClear,
}) {
  return (
    <>
      <PageHeading
        eyebrow="IMAGE WORKSPACE / 02"
        title="病灶图像分析"
        description="上传皮肤镜图像，调用已接入的七分类模型。"
      />

      <section className="analysis-layout">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">INPUT IMAGE</span>
              <h2>输入图像</h2>
            </div>
            <span className="step-label">STEP 01</span>
          </div>

          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept="image/*"
            onChange={onFileChange}
          />

          {previewUrl ? (
            <div className="image-preview">
              <img src={previewUrl} alt="已选择的皮肤镜图像预览" />

              <div className="image-file-info">
                <span className="file-check">✓</span>
                <div>
                  <strong>{file?.name}</strong>
                  <span>
                    {file
                      ? `${(file.size / 1024).toFixed(1)} KB`
                      : 'Image loaded'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                onFile(event.dataTransfer.files?.[0])
              }}
            >
              <span className="upload-icon">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v5h14v-5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>

              <strong>选择一张皮肤镜图像</strong>
              <span>点击选择，或将图片拖放到这里</span>
              <span className="file-format">支持常见图片格式，最大 10 MB</span>
            </button>
          )}

          <div className="upload-actions">
            <button
              type="button"
              className="button button-secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              {previewUrl ? '更换图片' : '选择图片'}
            </button>

            {previewUrl && (
              <button
                type="button"
                className="text-button"
                onClick={onClear}
                disabled={analyzing}
              >
                移除图片
              </button>
            )}
          </div>

          <div className="input-guidance">
            <span className="guidance-icon">i</span>
            <p>
              图片会发送至本机 Python 后端进行分类推理。请使用适合研究的数据图片。
            </p>
          </div>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">MODEL OUTPUT</span>
              <h2>分析结果</h2>
            </div>

            <span className={`pending-pill ${analysisResult ? 'connected' : ''}`}>
              <span className="status-dot" />
              {analysisResult ? '已完成' : analyzing ? '分析中' : '等待分析'}
            </span>
          </div>

          <div className={`result-preview ${previewUrl ? 'has-image' : ''}`}>
            {previewUrl ? (
              <img src={previewUrl} alt="原始输入图像" />
            ) : (
              <div className="empty-result">
                <span className="empty-result-icon">◎</span>
                <strong>等待输入图像</strong>
                <p>选择左侧图像后，可以在此预览输入内容。</p>
              </div>
            )}

            <span className="result-image-label">ORIGINAL IMAGE</span>
          </div>

          {analysisResult ? (
            <div className="classification-summary">
              <span className="panel-kicker">CLASSIFICATION RESULT</span>
              <div className="prediction-name">
                {analysisResult.predicted_class}
              </div>
              <div className="prediction-score">
                模型最高分数：{Number(
                  analysisResult.model_score_percentage,
                ).toFixed(2)}%
              </div>
              <p>
                权重版本：第 {analysisResult.model_epoch} 轮训练模型
              </p>
            </div>
          ) : (
            <div className="result-placeholders">
              <div>
                <span>01</span>
                <strong>七分类结果</strong>
                <small>{analyzing ? '正在推理……' : '等待模型推理'}</small>
              </div>
              <div>
                <span>02</span>
                <strong>分割掩码</strong>
                <small>待接入分割模型</small>
              </div>
            </div>
          )}

          <button
            type="button"
            className="button button-primary analyze-button"
            onClick={onAnalyze}
            disabled={analyzing || !file}
          >
            {analyzing ? '正在分析……' : '运行分类模型'}
            <span>{analyzing ? '…' : '→'}</span>
          </button>

          <p className="result-disclaimer">
            此操作运行真实的七分类模型；分割掩码和边界叠加功能尚未接入。
          </p>
        </div>
      </section>

      {message && (
        <div className="inline-message" role="status" aria-live="polite">
          <span>i</span>
          <p>{message}</p>
        </div>
      )}

      {analysisResult && (
        <ClassificationResult result={analysisResult} />
      )}

      <section className="workflow-strip">
        <div className="workflow-title">分析流程</div>

        <div className="workflow-step active">
          <span>01</span>
          <div>
            <strong>图像输入</strong>
            <small>{file ? '已载入' : '等待输入'}</small>
          </div>
        </div>

        <span className="workflow-line" />

        <div className={`workflow-step ${analysisResult ? 'active' : ''}`}>
          <span>02</span>
          <div>
            <strong>七分类推理</strong>
            <small>{analysisResult ? '已完成' : '可运行'}</small>
          </div>
        </div>

        <span className="workflow-line" />

        <div className="workflow-step">
          <span>03</span>
          <div>
            <strong>病灶分割与融合</strong>
            <small>待接入</small>
          </div>
        </div>
      </section>
    </>
  )
}

function ClassificationResult({ result }) {
  return (
    <section className="panel real-classification-result">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">REAL MODEL RESPONSE</span>
          <h2>七分类模型输出</h2>
        </div>
        <span className="step-label">
          EPOCH {result.model_epoch}
        </span>
      </div>

      <p className="result-file-name">
        输入图片：{result.filename}
      </p>

      <div className="score-list">
        {result.scores.map((item) => {
          const percentage = Number(item.percentage)

          return (
            <div className="score-row" key={item.code}>
              <span className="score-code">{item.code}</span>

              <div
                className="score-track"
                role="progressbar"
                aria-label={`${item.code} 模型分数`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percentage}
              >
                <div
                  className={`score-fill ${
                    item.code === result.predicted_class ? 'score-fill-best' : ''
                  }`}
                  style={{
                    width: `${Math.min(100, Math.max(0, percentage))}%`,
                  }}
                />
              </div>

              <span className="score-value">
                {percentage.toFixed(2)}%
              </span>
            </div>
          )
        })}
      </div>

      <div className="research-notice">
        <strong>科研用途提示</strong>
        <p>{result.notice}</p>
        <p>
          上述分数为模型 softmax 输出，并非经过临床校准的患病概率，也不能单独用于医疗诊断。
        </p>
      </div>
    </section>
  )
}

function Classification({ analysisResult, onStart }) {
  return (
    <>
      <PageHeading
        eyebrow="CLASSIFICATION / 03"
        title="分类与融合结果"
        description="查看已接入的七分类模型输出，以及后续分割引导分类融合的开发规划。"
        action={
          <button className="button button-primary" onClick={onStart}>
            新建分析 <span>↗</span>
          </button>
        }
      />

      {analysisResult ? (
        <>
          <section className="panel classification-overview">
            <span className="panel-kicker">LATEST CLASSIFICATION</span>
            <h2>最近一次模型结果</h2>

            <div className="classification-highlight">
              <div>
                <span className="metric-label">预测类别</span>
                <strong className="classification-class">
                  {analysisResult.predicted_class}
                </strong>
              </div>

              <div>
                <span className="metric-label">最高模型分数</span>
                <strong className="classification-percentage">
                  {Number(
                    analysisResult.model_score_percentage,
                  ).toFixed(2)}%
                </strong>
              </div>
            </div>

            <p className="result-file-name">
              图片：{analysisResult.filename}
            </p>

            <p className="result-file-name">
              模型：第 {analysisResult.model_epoch} 轮训练权重
            </p>
          </section>

          <ClassificationResult result={analysisResult} />
        </>
      ) : (
        <section className="panel placeholder-panel">
          <span className="placeholder-symbol">◈</span>
          <span className="panel-kicker">CLASSIFICATION OUTPUT</span>
          <h2>尚无预测结果</h2>
          <p>
            七分类模型后端已经接入。请先在病灶分析页面上传一张图片并运行模型，成功后即可在此查看真实的类别分数。
          </p>

          <button className="button button-primary" onClick={onStart}>
            前往病灶分析 <span>→</span>
          </button>

          <div className="planned-list">
            <div><span>01</span>七分类预测与模型分数</div>
            <div><span>02</span>后续增加基础分类与融合分类对比</div>
            <div><span>03</span>后续增加模型版本与推理记录</div>
          </div>

          <span className="large-status">CLASSIFIER API CONNECTED</span>
        </section>
      )}

      <section className="panel fusion-notice">
        <span className="panel-kicker">FUTURE DEVELOPMENT</span>
        <h2>分割引导分类融合</h2>
        <p>
          当前展示的是独立 RGB 七分类模型的结果，不是分割引导融合结果。融合模块需要在完成分割模型接入和融合方法实现后，再展示真实实验数据。
        </p>
      </section>
    </>
  )
}

function Evaluation() {
  return (
    <>
      <PageHeading
        eyebrow="EXPERIMENTS / 04"
        title="模型评估"
        description="统一查看分割、分类和融合实验的实际评估结果。"
      />

      <div className="metric-grid">
        {[
          ['Dice', '分割重叠度'],
          ['IoU', '交并比'],
          ['Macro-F1', '分类宏平均 F1'],
          ['Accuracy', '分类准确率'],
        ].map(([name, description]) => (
          <div className="metric-card" key={name}>
            <span className="metric-label">{name}</span>
            <strong>—</strong>
            <span className="metric-description">{description}</span>
            <span className="metric-pending">尚未载入评估数据</span>
          </div>
        ))}
      </div>

      <section className="panel evaluation-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">EXPERIMENT RECORDS</span>
            <h2>实验记录</h2>
          </div>
          <span className="pending-pill">暂无数据源</span>
        </div>

        <div className="empty-table">
          <span className="empty-result-icon">▥</span>
          <strong>尚未连接实验数据</strong>
          <p>
            接入真实评估文件后，这里可以展示实验名称、数据集划分、模型版本和指标。
          </p>
        </div>
      </section>

      <p className="evaluation-note">
        指标暂时显示为“—”，不会使用虚构数值。后续将根据真实评估脚本输出接入数据。
      </p>
    </>
  )
}

export default App
