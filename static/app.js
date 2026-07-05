document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('agent-form');
  const requestInput = document.getElementById('request-input');
  const submitBtn = document.getElementById('submit-btn');
  const errorMsg = document.getElementById('error-message');

  const stateIdle = document.getElementById('state-idle');
  const stateProcessing = document.getElementById('state-processing');
  const stateResult = document.getElementById('state-result');

  // Steps
  const stepUnderstood = document.getElementById('step-understood');
  const stepPlanCreated = document.getElementById('step-plan-created');
  const stepContentGenerated = document.getElementById('step-content-generated');
  const stepReviewComplete = document.getElementById('step-review-complete');
  const stepDocxCreated = document.getElementById('step-docx-created');

  // Result Elements
  const resDocType = document.getElementById('res-doc-type');
  const resTitle = document.getElementById('res-title');
  const resAssumptions = document.getElementById('res-assumptions');
  const resPlanList = document.getElementById('res-plan-list');
  const resTaskList = document.getElementById('res-task-list');
  const resReviewNotes = document.getElementById('res-review-notes');
  const resDownloadBtn = document.getElementById('res-download-btn');
  const resViewBtn = document.getElementById('res-view-btn');
  const resFileName = document.getElementById('res-file-name');
  const resTotalTime = document.getElementById('res-total-time');

  // Time Metrics Elements (Success State)
  const timePlanning = document.getElementById('time-planning');
  const timeExecution = document.getElementById('time-execution');
  const timeReflection = document.getElementById('time-reflection');
  const timeDocument = document.getElementById('time-document');

  // Live Timer Elements (Processing State)
  const liveTimePlanning = document.getElementById('live-time-planning');
  const liveTimeExecution = document.getElementById('live-time-execution');
  const liveTimeReflection = document.getElementById('live-time-reflection');
  const liveTimeDocument = document.getElementById('live-time-document');
  const liveTimeTotal = document.getElementById('live-time-total');

  // Timers, intervals, and current state URLs
  let progressTimeouts = [];
  let liveTimerInterval = null;
  
  let startTime = 0;
  let planningStart = null;
  let planningEnd = null;
  let executionStart = null;
  let executionEnd = null;
  let reflectionStart = null;
  let reflectionEnd = null;
  let documentStart = null;
  let documentEnd = null;

  let currentDownloadUrl = null;
  let currentDocTitle = "";

  function showError(msg) {
    errorMsg.innerText = msg;
    errorMsg.classList.remove('hidden');
  }

  function clearError() {
    errorMsg.innerText = '';
    errorMsg.classList.add('hidden');
  }

  function resetSteps() {
    const steps = [stepUnderstood, stepPlanCreated, stepContentGenerated, stepReviewComplete, stepDocxCreated];
    steps.forEach(step => {
      step.className = 'progress-step';
    });
    
    // Clear timeouts and intervals
    progressTimeouts.forEach(clearTimeout);
    progressTimeouts = [];
    if (liveTimerInterval) {
      clearInterval(liveTimerInterval);
      liveTimerInterval = null;
    }
    
    // Reset times variables
    startTime = 0;
    planningStart = null;
    planningEnd = null;
    executionStart = null;
    executionEnd = null;
    reflectionStart = null;
    reflectionEnd = null;
    documentStart = null;
    documentEnd = null;

    currentDownloadUrl = null;
    currentDocTitle = "";

    // Reset live text displays
    const liveDisplays = [liveTimePlanning, liveTimeExecution, liveTimeReflection, liveTimeDocument, liveTimeTotal];
    liveDisplays.forEach(disp => disp.innerText = '0.00');
  }

  function setStepState(stepElement, state) {
    stepElement.className = 'progress-step';
    if (state === 'active') {
      stepElement.classList.add('active');
    } else if (state === 'completed') {
      stepElement.classList.add('completed');
    }
  }

  // Update Live Stopwatches
  function updateLiveTimers() {
    const now = performance.now();
    
    // Total Elapsed
    liveTimeTotal.innerText = ((now - startTime) / 1000).toFixed(2);
    
    // Planning Timer
    if (planningStart !== null) {
      if (planningEnd === null) {
        liveTimePlanning.innerText = ((now - planningStart) / 1000).toFixed(2);
      } else {
        liveTimePlanning.innerText = ((planningEnd - planningStart) / 1000).toFixed(2);
      }
    }
    
    // Execution Timer
    if (executionStart !== null) {
      if (executionEnd === null) {
        liveTimeExecution.innerText = ((now - executionStart) / 1000).toFixed(2);
      } else {
        liveTimeExecution.innerText = ((executionEnd - executionStart) / 1000).toFixed(2);
      }
    }
    
    // Reflection Timer
    if (reflectionStart !== null) {
      if (reflectionEnd === null) {
        liveTimeReflection.innerText = ((now - reflectionStart) / 1000).toFixed(2);
      } else {
        liveTimeReflection.innerText = ((reflectionEnd - reflectionStart) / 1000).toFixed(2);
      }
    }
    
    // Document Timer
    if (documentStart !== null) {
      if (documentEnd === null) {
        liveTimeDocument.innerText = ((now - documentStart) / 1000).toFixed(2);
      } else {
        liveTimeDocument.innerText = ((documentEnd - documentStart) / 1000).toFixed(2);
      }
    }
  }

  // Simulates progress steps with timed activation
  function startProgressSimulation() {
    resetSteps();
    
    startTime = performance.now();
    
    // Step 1: Request Understood active immediately
    setStepState(stepUnderstood, 'active');
    planningStart = performance.now();

    // Start Live Clock
    liveTimerInterval = setInterval(updateLiveTimers, 50);

    // Step 2: Plan Created active after 1.5s
    progressTimeouts.push(setTimeout(() => {
      setStepState(stepUnderstood, 'completed');
      setStepState(stepPlanCreated, 'active');
      planningEnd = performance.now();
      executionStart = performance.now();
    }, 1500));

    // Step 3: Content Generated active after 3.5s
    progressTimeouts.push(setTimeout(() => {
      setStepState(stepPlanCreated, 'completed');
      setStepState(stepContentGenerated, 'active');
      executionEnd = performance.now();
      reflectionStart = performance.now();
    }, 3500));

    // Step 4: Self Review Complete active after 6.5s
    progressTimeouts.push(setTimeout(() => {
      setStepState(stepContentGenerated, 'completed');
      setStepState(stepReviewComplete, 'active');
      reflectionEnd = performance.now();
      documentStart = performance.now();
    }, 6500));

    // Step 5: DOCX Created active after 8.5s
    progressTimeouts.push(setTimeout(() => {
      setStepState(stepReviewComplete, 'completed');
      setStepState(stepDocxCreated, 'active');
      documentEnd = performance.now();
    }, 8500));
  }

  function completeAllSteps() {
    // Stop simulation timeouts & live clock
    progressTimeouts.forEach(clearTimeout);
    progressTimeouts = [];
    if (liveTimerInterval) {
      clearInterval(liveTimerInterval);
      liveTimerInterval = null;
    }
    
    setStepState(stepUnderstood, 'completed');
    setStepState(stepPlanCreated, 'completed');
    setStepState(stepContentGenerated, 'completed');
    setStepState(stepReviewComplete, 'completed');
    setStepState(stepDocxCreated, 'completed');
  }

  // View Document preview handler without download trigger
  resViewBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (!currentDownloadUrl) return;

    const originalContent = resViewBtn.innerHTML;
    resViewBtn.innerHTML = '<span>Loading Preview...</span>';

    try {
      const response = await fetch(currentDownloadUrl);
      const arrayBuffer = await response.arrayBuffer();
      
      mammoth.convertToHtml({ arrayBuffer: arrayBuffer })
        .then(function(result) {
          const html = result.value;
          
          // Open a new tab and write the rendered HTML
          const newWindow = window.open();
          if (newWindow) {
            newWindow.document.write(`
              <!DOCTYPE html>
              <html>
              <head>
                <title>${currentDocTitle || "Document Preview"}</title>
                <style>
                  body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 3rem auto;
                    padding: 0 1.5rem;
                    color: #111827;
                    background-color: #ffffff;
                  }
                  h1 {
                    font-size: 2.2rem;
                    font-weight: 800;
                    letter-spacing: -0.025em;
                    color: #111827;
                    border-bottom: 2px solid #f3f4f6;
                    padding-bottom: 0.75rem;
                    margin-top: 0;
                    margin-bottom: 1.5rem;
                    text-align: center;
                  }
                  h2 {
                    font-size: 1.5rem;
                    font-weight: 700;
                    letter-spacing: -0.02em;
                    color: #1f2937;
                    margin-top: 2rem;
                    margin-bottom: 0.75rem;
                    border-bottom: 1px solid #f3f4f6;
                    padding-bottom: 0.5rem;
                  }
                  p {
                    margin-bottom: 1.25rem;
                    color: #374151;
                    font-size: 1.05rem;
                  }
                  ul, ol {
                    margin-bottom: 1.25rem;
                    padding-left: 2rem;
                    color: #374151;
                    font-size: 1.05rem;
                  }
                  li {
                    margin-bottom: 0.4rem;
                  }
                  strong {
                    color: #111827;
                  }
                </style>
              </head>
              <body>
                ${html}
              </body>
              </html>
            `);
            newWindow.document.close();
          } else {
            alert("Popup blocker blocked the preview window. Please allow popups for this site.");
          }
        })
        .catch(function(err) {
          console.error(err);
          alert("Failed to render the document preview.");
        });
    } catch (error) {
      console.error(error);
      alert("Could not load the document file. Please ensure it is downloadable.");
    } finally {
      resViewBtn.innerHTML = originalContent;
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();

    const requestText = requestInput.value.trim();
    if (!requestText) return;

    // UI State: disabled inputs, show processing panel
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Processing Agent...</span>';
    
    stateIdle.classList.add('hidden');
    stateResult.classList.add('hidden');
    stateProcessing.classList.remove('hidden');

    startProgressSimulation();

    try {
      const response = await fetch('/agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ request: requestText })
      });

      const data = await response.json();

      if (!response.ok) {
        const msg = data.detail || 'LLM unavailable. Please try again in a few seconds.';
        throw new Error(msg);
      }

      // Success path
      completeAllSteps();
      
      // Delay slightly for the user to see the completed state
      setTimeout(() => {
        // Populate Success Banner Title and Time
        resTotalTime.innerText = data.times.total;
        
        // Populate Times Grid
        timePlanning.innerText = data.times.planning;
        timeExecution.innerText = data.times.execution;
        timeReflection.innerText = data.times.reflection;
        timeDocument.innerText = data.times.document;

        // Populate Result Title & Document Type
        resDocType.innerText = data.document_type;
        // Extracts the display title name from message
        resTitle.innerText = data.message.replace("Successfully planned, drafted, revised, and generated the document ", "").replace(".", "").replace(/'/g, "");
        
        // Assumptions list
        resAssumptions.innerHTML = '';
        if (data.assumptions && data.assumptions.length > 0) {
          data.assumptions.forEach(ass => {
            const li = document.createElement('li');
            li.innerText = ass;
            resAssumptions.appendChild(li);
          });
        } else {
          const li = document.createElement('li');
          li.innerText = 'No assumptions were required.';
          resAssumptions.appendChild(li);
        }

        // Agent Task List checklist
        resTaskList.innerHTML = '';
        if (data.agent_tasks && data.agent_tasks.length > 0) {
          data.agent_tasks.forEach(task => {
            const li = document.createElement('li');
            li.innerText = task;
            resTaskList.appendChild(li);
          });
        }

        // Document Outline list
        resPlanList.innerHTML = '';
        if (data.plan && data.plan.length > 0) {
          data.plan.forEach(sec => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${sec.heading}</strong>: ${sec.purpose}`;
            resPlanList.appendChild(li);
          });
        }

        // Review notes (Self Check Panel)
        resReviewNotes.innerHTML = '';
        if (data.review_notes && data.review_notes.length > 0) {
          data.review_notes.forEach(note => {
            const li = document.createElement('li');
            li.innerText = note;
            resReviewNotes.appendChild(li);
          });
        } else {
          const li = document.createElement('li');
          li.innerText = 'All sections passed evaluation on the first attempt with no errors.';
          resReviewNotes.appendChild(li);
        }

        // Setup current URL references for the view handler
        currentDownloadUrl = data.download_url;
        currentDocTitle = resTitle.innerText;

        // Action download link & Filename
        const fname = data.download_url.split('/').pop();
        resFileName.innerText = fname;
        resDownloadBtn.href = data.download_url;

        // Transition views
        stateProcessing.classList.add('hidden');
        stateResult.classList.remove('hidden');
      }, 500);

    } catch (err) {
      completeAllSteps();
      stateProcessing.classList.add('hidden');
      stateIdle.classList.remove('hidden');
      showError(err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>Generate Document</span>';
    }
  });
});
