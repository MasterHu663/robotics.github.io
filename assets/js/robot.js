/**
 * Robot Interaction Controller
 * Handles clicking to toggle the waving animation
 */
document.addEventListener('DOMContentLoaded', () => {
    const robotElement = document.getElementById('main-robot');

    if (robotElement) {
        robotElement.addEventListener('click', () => {
            // Toggle the waving class
            robotElement.classList.toggle('is-waving');

            // Logic: Auto-stop waving after 2 seconds for better UX
            if (robotElement.classList.contains('is-waving')) {
                setTimeout(() => {
                    robotElement.classList.remove('is-waving');
                }, 2000);
            }
        });
    }
});