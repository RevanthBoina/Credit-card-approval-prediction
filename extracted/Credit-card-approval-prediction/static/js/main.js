/**
 * Keep numeric applicant form fields within realistic visible ranges.
 */
function clampNumericInputs() {
    document.querySelectorAll('input[type="number"]').forEach((input) => {
        input.addEventListener("blur", () => {
            const min = input.min === "" ? null : Number(input.min);
            const max = input.max === "" ? null : Number(input.max);
            let value = Number(input.value);
            if (Number.isNaN(value)) {
                return;
            }
            if (min !== null && value < min) {
                value = min;
            }
            if (max !== null && value > max) {
                value = max;
            }
            input.value = value;
        });
    });
}

document.addEventListener("DOMContentLoaded", clampNumericInputs);

/**
 * Clear assessment forms when the prediction page is opened from result/back navigation.
 */
function clearAssessmentForms() {
    document.querySelectorAll('form[data-clear-on-load="true"]').forEach((form) => {
        form.reset();
        form.querySelectorAll("select").forEach((select) => {
            select.selectedIndex = 0;
        });
    });
}

window.addEventListener("pageshow", clearAssessmentForms);
