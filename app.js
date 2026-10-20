const API_BASE_URL = "http://127.0.0.1:8001";

document.addEventListener("DOMContentLoaded", () => {
    const landingState = document.getElementById("state-landing");
    const formState = document.getElementById("state-form");
    const resultState = document.getElementById("state-result");

    const btnStart = document.getElementById("btn-start");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnNew = document.getElementById("btn-new");
    const btnReturn = document.getElementById("btn-return");

    const form = document.getElementById("analysis-form");
    const loadingIndicator = document.getElementById("loading-indicator");
    const errorMsg = document.getElementById("error-message");

    const resultLabel = document.getElementById("result-label");
    const resultScore = document.getElementById("result-score");

    // Navigation logic
    function showState(state) {
        landingState.style.display = "none";
        formState.style.display = "none";
        resultState.style.display = "none";

        if (state === "landing") landingState.style.display = "flex";
        if (state === "form") formState.style.display = "flex";
        if (state === "result") resultState.style.display = "flex";
    }

    btnStart.addEventListener("click", () => {
        showState("form");
    });

    btnReturn.addEventListener("click", () => {
        showState("landing");
    });

    btnNew.addEventListener("click", () => {
        form.reset();
        errorMsg.style.display = "none";
        showState("form");
    });

    // Form submission logic
    btnAnalyze.addEventListener("click", async () => {
        // Validate required
        const titleInput = document.getElementById("job-title").value.trim();
        if (!titleInput) {
            showError("Job Title is required.");
            return;
        }

        // Construct payload
        const payload = {
            title: titleInput,
            location: document.getElementById("job-location").value.trim() || null,
            department: document.getElementById("job-department").value.trim() || null,
            salary_range: document.getElementById("job-salary").value.trim() || null,
            company_profile: document.getElementById("job-company-profile").value.trim() || null,
            description: document.getElementById("job-description").value.trim() || null,
            requirements: document.getElementById("job-requirements").value.trim() || null,
            benefits: document.getElementById("job-benefits").value.trim() || null,
            telecommuting: document.getElementById("job-telecommuting").checked ? 1 : 0,
            has_company_logo: document.getElementById("job-company-logo").checked ? 1 : 0,
            has_questions: document.getElementById("job-has-questions").checked ? 1 : 0,
            employment_type: document.getElementById("job-employment-type").value || null,
            required_experience: document.getElementById("job-experience").value || null,
            required_education: document.getElementById("job-education").value || null,
            industry: document.getElementById("job-industry").value.trim() || null,
            function: document.getElementById("job-function").value.trim() || null,
        };

        // UI Loading State
        btnAnalyze.disabled = true;
        btnAnalyze.textContent = "Analyzing...";
        loadingIndicator.style.display = "block";
        errorMsg.style.display = "none";

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Show result
            displayResult(data);
            showState("result");

        } catch (err) {
            console.error("Prediction error:", err);
            showError("Unable to analyze this posting right now. Please try again.");
        } finally {
            // Restore UI
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = "Analyze Job";
            loadingIndicator.style.display = "none";
        }
    });

    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.style.display = "block";
    }

    function displayResult(data) {
        resultLabel.textContent = data.prediction;
        // Match existing aesthetic
        if (data.fraudulent === 1) {
            resultLabel.style.color = "#ff4a4a"; // subtle red for fraudulent
        } else {
            resultLabel.style.color = "#e8e8e8";
        }

        resultScore.textContent = `Model Decision Score: ${data.decision_score.toFixed(4)}`;
    }
});
