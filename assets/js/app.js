/**
 * Robotic Portfolio Core Logic
 * Clean and lightweight
 */
document.addEventListener('DOMContentLoaded', () => {
    const typewriter = (id, text) => {
        const el = document.getElementById(id);
        if (!el) return;
        
        let i = 0;
        const type = () => {
            if (i < text.length) {
                el.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, 100);
            }
        };
        type();
    };

    // 执行打字效果
    typewriter('typewriter', 'SYSTEMS_ENGINEER_ACTIVE');
});