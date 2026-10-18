import { useRef, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/api/predict'
const SEGMENT_API_URL = 'http://127.0.0.1:8000/api/segment'
const MAX_FILE_SIZE = 10 * 1024 * 1024

const navigation = [
  { id: 'overview', label: 'Workspace', number: '01', icon: '⌂' },
  { id: 'analysis', label: 'Lesion Analysis', number: '02', icon: '◎' },
  { id: 'classification', label: 'Classification', number: '03', icon: '◈' },
  { id: 'evaluation', label: 'Model Evaluation', number: '04', icon: '▥' },
]

function App() {
  const [activePage, setActivePage] = useState('overview')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [message, setMessage] = useState('')
  const [dragging, setDragging] = useState(false)

  const [analysisResult, setAnalysisResult] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  const [segmentationResult, setSegmentationResult] = useState(null)
  const [segmenting, setSegmenting] = useState(false)

  const fileInputRef = useRef(null)

  const currentPage =
    navigation.find((item) => item.id === activePage) ?? navigation[0]

  function handleFile(file) {
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setMessage('Please select an image file (JPG, PNG, etc.).')
      return
    }

    if (file.size > MAX_FILE_SIZE) {
      setMessage('Image size must not exceed 10 MB. Please choose a smaller image.')
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))

    // Clear previous classification and segmentation results when the image changes.
    setAnalysisResult(null)
    setSegmentationResult(null)

    setMessage('Image loaded. You can now run the classification and segmentation models.')
    setActivePage('analysis')
  }

  function handleFileChange(event) {
    handleFile(event.target.files?.[0])
    event.target.value = ''
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setMessage('Please upload a dermoscopic image first.')
      return
    }

    if (analyzing) return

    setAnalyzing(true)
    setAnalysisResult(null)
    setMessage('Sending the image to the classification model, please wait...')

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
        throw new Error('The backend response could not be parsed. Please check the API service.')
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : `Request failed, HTTP status: ${response.status}`,
        )
      }

      if (
        !data.predicted_class ||
        !Array.isArray(data.scores) ||
        data.scores.length !== 7
      ) {
        throw new Error('The backend response format is not as expected.')
      }

      setAnalysisResult(data)
      setMessage('Classification inference complete. The results below come from the loaded model.')
    } catch (error) {
      console.error('Classification request failed:', error)

      setAnalysisResult(null)

      if (error instanceof TypeError) {
        setMessage(
          'Unable to connect to the Python backend. Make sure the service on port 8000 is running, then try again.',
        )
      } else {
        setMessage(`Classification failed: ${error.message}`)
      }
    } finally {
      setAnalyzing(false)
    }
  }

  async function handleSegment() {
    if (!selectedFile) {
      setMessage('Please upload a dermoscopic image first.')
      return
    }

    if (segmenting) return

    setSegmenting(true)
    setSegmentationResult(null)
    setMessage('Calling the MALUNet segmentation model, please wait...')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(SEGMENT_API_URL, {
        method: 'POST',
        body: formData,
      })

      let data

      try {
        data = await response.json()
      } catch {
        throw new Error('The backend response could not be parsed. Please check the API service.')
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : `Request failed, HTTP status: ${response.status}`,
        )
      }

      if (
        typeof data.mask !== 'string' ||
        typeof data.overlay !== 'string'
      ) {
        throw new Error('The backend did not return a valid segmentation mask or overlay image.')
      }

      setSegmentationResult(data)
      setMessage('MALUNet segmentation complete. Results are displayed below.')
    } catch (error) {
      console.error('Segmentation request failed:', error)

      setSegmentationResult(null)

      if (error instanceof TypeError) {
        setMessage(
          'Unable to connect to the Python backend. Make sure the backend service is running, then try again.',
        )
      } else {
        setMessage(`Segmentation failed: ${error.message}`)
      }
    } finally {
      setSegmenting(false)
    }
  }

  function handleClear() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setPreviewUrl('')
    setSelectedFile(null)
    setAnalysisResult(null)
    setSegmentationResult(null)
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

        <nav className="side-nav" aria-label="Primary navigation">
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
            CLASSIFIER + SEGMENTATION
          </div>

          <p>Weak Boundary Guided Skin Lesion Detection System</p>

          <span className="project-version">
            Research prototype · v0.3
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
              onSegment={handleSegment}
              segmenting={segmenting}
              segmentationResult={segmentationResult}
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
        title="Lesion Analysis Workspace"
        description="Explore skin lesion segmentation, boundary analysis, and classification methods."
        action={
          <button className="button button-primary" onClick={onStart}>
            <span>＋</span> New Analysis
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
            Start from images,
            <br />
            <span>understand lesion boundaries.</span>
          </h2>

          <p>
            A research prototype for skin lesion image analysis, progressively integrating segmentation, classification, and model evaluation workflows.
          </p>

          <button className="button button-light" onClick={onStart}>
            Start Image Analysis <span>↗</span>
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
          <h2>Research Modules</h2>
          <p>Features built incrementally around the thesis research goals</p>
        </div>
        <span className="section-note">04 MODULES</span>
      </div>

      <section className="module-grid">
        <ModuleCard
          number="01"
          icon="◎"
          title="Lesion Segmentation"
          description="Uses MALUNet to display predicted masks and boundary overlay results."
          status="MALUNet segmentation API connected"
          onClick={() => onNavigate('analysis')}
        />

        <ModuleCard
          number="02"
          icon="◈"
          title="Feature Classification"
          description="Uses the connected 7-class model to output real prediction scores."
          status="Classification model connected"
          onClick={() => onNavigate('classification')}
        />

        <ModuleCard
          number="03"
          icon="⌘"
          title="Segmentation-Guided Fusion"
          description="Reserves an interface for comparing baseline and fusion classification."
          status="Planned"
          onClick={() => onNavigate('classification')}
        />

        <ModuleCard
          number="04"
          icon="▥"
          title="Experiment Evaluation"
          description="Displays Dice, IoU, and classification metrics from real experiments."
          status="Evaluation data pending"
          onClick={() => onNavigate('evaluation')}
        />
      </section>

      <section className="notice-panel">
        <div className="notice-icon">i</div>
        <div>
          <strong>Current System Progress</strong>
          <p>
            The 7-class model and MALUNet segmentation API are now callable from the frontend. Segmentation-guided fusion and real experiment evaluation data still require further development and validation.
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
  onSegment,
  segmenting,
  segmentationResult,
  message,
  onClear,
}) {
  return (
    <>
      <PageHeading
        eyebrow="IMAGE WORKSPACE / 02"
        title="Lesion Image Analysis"
        description="Upload a dermoscopic image and run the 7-class model and MALUNet segmentation model separately."
      />

      <section className="analysis-layout">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">INPUT IMAGE</span>
              <h2>Input Image</h2>
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
              <img src={previewUrl} alt="Preview of the selected dermoscopic image" />

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

              <strong>Select a dermoscopic image</strong>
              <span>Click to browse, or drag and drop an image here</span>
              <span className="file-format">
                Common image formats supported, up to 10 MB
              </span>
            </button>
          )}

          <div className="upload-actions">
            <button
              type="button"
              className="button button-secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              {previewUrl ? 'Change Image' : 'Select Image'}
            </button>

            {previewUrl && (
              <button
                type="button"
                className="text-button"
                onClick={onClear}
                disabled={analyzing || segmenting}
              >
                Remove Image
              </button>
            )}
          </div>

          <div className="input-guidance">
            <span className="guidance-icon">i</span>
            <p>
              Images are sent to the local Python backend for model inference. Please use images appropriate for research.
            </p>
          </div>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">MODEL OUTPUT</span>
              <h2>Analysis Results</h2>
            </div>

            <span
              className={`pending-pill ${
                analysisResult || segmentationResult ? 'connected' : ''
              }`}
            >
              <span className="status-dot" />
              {analyzing || segmenting
                ? 'Analyzing'
                : analysisResult || segmentationResult
                  ? 'Complete'
                  : 'Awaiting Analysis'}
            </span>
          </div>

          <div className={`result-preview ${previewUrl ? 'has-image' : ''}`}>
            {previewUrl ? (
              <img src={previewUrl} alt="Original input image" />
            ) : (
              <div className="empty-result">
                <span className="empty-result-icon">◎</span>
                <strong>Waiting for Input Image</strong>
                <p>After selecting an image on the left, you can preview it here.</p>
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
                Top model score:
                {Number(analysisResult.model_score_percentage).toFixed(2)}%
              </div>
              <p>
                Weights: model from training epoch {analysisResult.model_epoch}
              </p>
            </div>
          ) : (
            <div className="result-placeholders">
              <div>
                <span>01</span>
                <strong>7-Class Result</strong>
                <small>
                  {analyzing ? 'Running inference...' : 'Awaiting model inference'}
                </small>
              </div>
              <div>
                <span>02</span>
                <strong>Segmentation Mask</strong>
                <small>
                  {segmenting ? 'Segmenting...' : 'Awaiting MALUNet inference'}
                </small>
              </div>
            </div>
          )}

          <button
            type="button"
            className="button button-primary analyze-button"
            onClick={onAnalyze}
            disabled={analyzing || segmenting || !file}
          >
            {analyzing ? 'Analyzing...' : 'Run Classification Model'}
            <span>{analyzing ? '…' : '→'}</span>
          </button>

          <button
            type="button"
            className="button button-secondary analyze-button"
            onClick={onSegment}
            disabled={segmenting || analyzing || !file}
          >
            {segmenting ? 'Segmenting...' : 'Run MALUNet Segmentation Model'}
            <span>{segmenting ? '…' : '→'}</span>
          </button>

          <p className="result-disclaimer">
            Classification and segmentation are two independent inference processes. Running one model does not automatically run the other.
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

      {segmentationResult && (
        <SegmentationResult result={segmentationResult} />
      )}

      <section className="workflow-strip">
        <div className="workflow-title">Analysis Workflow</div>

        <div className="workflow-step active">
          <span>01</span>
          <div>
            <strong>Image Input</strong>
            <small>{file ? 'Loaded' : 'Awaiting input'}</small>
          </div>
        </div>

        <span className="workflow-line" />

        <div className={`workflow-step ${analysisResult ? 'active' : ''}`}>
          <span>02</span>
          <div>
            <strong>7-Class Inference</strong>
            <small>{analysisResult ? 'Complete' : 'Can run independently'}</small>
          </div>
        </div>

        <span className="workflow-line" />

        <div
          className={`workflow-step ${
            segmentationResult ? 'active' : ''
          }`}
        >
          <span>03</span>
          <div>
            <strong>MALUNet Lesion Segmentation</strong>
            <small>{segmentationResult ? 'Complete' : 'Can run independently'}</small>
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
          <h2>7-Class Model Output</h2>
        </div>
        <span className="step-label">
          EPOCH {result.model_epoch}
        </span>
      </div>

      <p className="result-file-name">
        Input image: {result.filename}
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
                aria-label={`${item.code} model score`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percentage}
              >
                <div
                  className={`score-fill ${
                    item.code === result.predicted_class
                      ? 'score-fill-best'
                      : ''
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
        <strong>Research Use Notice</strong>
        <p>{result.notice}</p>
        <p>
          The scores above are raw softmax outputs of the model, not clinically calibrated probabilities, and cannot be used alone for medical diagnosis.
        </p>
      </div>
    </section>
  )
}

function SegmentationResult({ result }) {
  return (
    <section className="panel real-classification-result segmentation-results">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">MALUNET SEGMENTATION</span>
          <h2>Lesion Segmentation Results</h2>
        </div>
        <span className="pending-pill connected">Segmentation Complete</span>
      </div>

      <p className="result-file-name">
        Input image: {result.filename}
      </p>

      <div className="segmentation-image-grid">
        <div className="segmentation-image-card">
          <h3>Predicted Mask</h3>
          <img
            src={result.mask}
            alt="Black-and-white lesion segmentation mask predicted by MALUNet"
          />
          <p>
            White areas indicate the model-predicted lesion region; black areas indicate other regions.
          </p>
        </div>

        <div className="segmentation-image-card">
          <h3>Boundary Overlay</h3>
          <img
            src={result.overlay}
            alt="Red lesion segmentation region and boundary overlaid on the original image"
          />
          <p>
            The red region visualizes the model prediction and does not represent a manually confirmed ground-truth boundary.
          </p>
        </div>
      </div>

      <div className="segmentation-metrics">
        <div>
          <span className="metric-label">Predicted Region Ratio</span>
          <strong>
            {Number(result.foreground_ratio).toFixed(2)}%
          </strong>
        </div>

        <div>
          <span className="metric-label">Input Size</span>
          <strong>
            {Array.isArray(result.input_size)
              ? result.input_size.join(' × ')
              : '256 × 256'}
          </strong>
        </div>

        <div>
          <span className="metric-label">Binarization Threshold</span>
          <strong>
            {Number(result.threshold).toFixed(2)}
          </strong>
        </div>
      </div>

      <div className="research-notice">
        <strong>Research Use Notice</strong>
        <p>
          {result.notice ||
            'Segmentation results are for research prototype demonstration only and cannot be used for medical diagnosis.'}
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
        title="Classification & Fusion Results"
        description="View the connected 7-class model output and the development plan for segmentation-guided classification fusion."
        action={
          <button className="button button-primary" onClick={onStart}>
            New Analysis <span>↗</span>
          </button>
        }
      />

      {analysisResult ? (
        <>
          <section className="panel classification-overview">
            <span className="panel-kicker">LATEST CLASSIFICATION</span>
            <h2>Latest Model Result</h2>

            <div className="classification-highlight">
              <div>
                <span className="metric-label">Predicted Class</span>
                <strong className="classification-class">
                  {analysisResult.predicted_class}
                </strong>
              </div>

              <div>
                <span className="metric-label">Top Model Score</span>
                <strong className="classification-percentage">
                  {Number(
                    analysisResult.model_score_percentage,
                  ).toFixed(2)}%
                </strong>
              </div>
            </div>

            <p className="result-file-name">
              Image: {analysisResult.filename}
            </p>

            <p className="result-file-name">
              Model: weights from training epoch {analysisResult.model_epoch}
            </p>
          </section>

          <ClassificationResult result={analysisResult} />
        </>
      ) : (
        <section className="panel placeholder-panel">
          <span className="placeholder-symbol">◈</span>
          <span className="panel-kicker">CLASSIFICATION OUTPUT</span>
          <h2>No Predictions Yet</h2>
          <p>
            The 7-class model backend is connected. Upload an image on the Lesion Analysis page and run the model; real class scores will appear here once it succeeds.
          </p>

          <button className="button button-primary" onClick={onStart}>
            Go to Lesion Analysis <span>→</span>
          </button>

          <div className="planned-list">
            <div><span>01</span>7-class prediction and model scores</div>
            <div><span>02</span>Add baseline vs. fusion classification comparison later</div>
            <div><span>03</span>Add model versions and inference records later</div>
          </div>

          <span className="large-status">CLASSIFIER API CONNECTED</span>
        </section>
      )}

      <section className="panel fusion-notice">
        <span className="panel-kicker">FUTURE DEVELOPMENT</span>
        <h2>Segmentation-Guided Classification Fusion</h2>
        <p>
          The results currently shown come from the standalone RGB 7-class model, not the segmentation-guided fusion. The fusion module will display real experimental data after the segmentation model integration and the fusion method are implemented.
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
        title="Model Evaluation"
        description="View actual evaluation results for segmentation, classification, and fusion experiments in one place."
      />

      <div className="metric-grid">
        {[
          ['Dice', 'Segmentation overlap'],
          ['IoU', 'Intersection over Union'],
          ['Macro-F1', 'Macro-averaged F1 for classification'],
          ['Accuracy', 'Classification accuracy'],
        ].map(([name, description]) => (
          <div className="metric-card" key={name}>
            <span className="metric-label">{name}</span>
            <strong>—</strong>
            <span className="metric-description">{description}</span>
            <span className="metric-pending">Evaluation data not loaded yet</span>
          </div>
        ))}
      </div>

      <section className="panel evaluation-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">EXPERIMENT RECORDS</span>
            <h2>Experiment Records</h2>
          </div>
          <span className="pending-pill">No Data Source</span>
        </div>

        <div className="empty-table">
          <span className="empty-result-icon">▥</span>
          <strong>No Experiment Data Connected</strong>
          <p>
            Once real evaluation files are connected, experiment names, dataset splits, model versions, and metrics will be shown here.
          </p>
        </div>
      </section>

      <p className="evaluation-note">
        Metrics are temporarily shown as "—"; no fabricated values are used. Data will be connected from real evaluation script outputs later.
      </p>
    </>
  )
}

export default App