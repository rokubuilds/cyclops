// Counter Animation for Stats
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = Math.ceil(target);
            clearInterval(timer);
        } else {
            element.textContent = Math.ceil(current);
        }
    }, 16);
}

// Initialize stat counters
function initStatCounters() {
    const statValues = document.querySelectorAll('.stat-value');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseFloat(entry.target.dataset.target);
                animateCounter(entry.target, target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statValues.forEach(stat => observer.observe(stat));
}

// Gait Visualization Canvas
class GaitVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        this.dataPoints = [];
        this.maxPoints = 600;
        this.time = 0;
        
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    generateDataPoint() {
        // Simulate gait cycle data
        const t = this.time;
        const amplitude = this.height / 3;
        const frequency = 0.05;
        
        const leftLeg = Math.sin(t * frequency) * amplitude + this.height / 2;
        const rightLeg = Math.sin(t * frequency + Math.PI) * amplitude + this.height / 2;
        
        return {
            time: t,
            leftLeg,
            rightLeg,
            x: this.width
        };
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw grid lines
        this.ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i < this.height; i += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, i);
            this.ctx.lineTo(this.width, i);
            this.ctx.stroke();
        }

        if (this.dataPoints.length < 2) return;

        // Draw left leg line
        this.ctx.strokeStyle = '#00d4ff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.dataPoints[0].x, this.dataPoints[0].leftLeg);
        
        for (let i = 1; i < this.dataPoints.length; i++) {
            this.ctx.lineTo(this.dataPoints[i].x, this.dataPoints[i].leftLeg);
        }
        this.ctx.stroke();

        // Draw right leg line
        this.ctx.strokeStyle = '#7c3aed';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.dataPoints[0].x, this.dataPoints[0].rightLeg);
        
        for (let i = 1; i < this.dataPoints.length; i++) {
            this.ctx.lineTo(this.dataPoints[i].x, this.dataPoints[i].rightLeg);
        }
        this.ctx.stroke();

        // Draw glow effect
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = '#00d4ff';
        this.ctx.strokeStyle = '#00d4ff';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        const lastPoint = this.dataPoints[this.dataPoints.length - 1];
        this.ctx.arc(lastPoint.x, lastPoint.leftLeg, 4, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }

    update() {
        // Add new data point
        const newPoint = this.generateDataPoint();
        this.dataPoints.push(newPoint);

        // Move all points left
        this.dataPoints.forEach(point => {
            point.x -= 2;
        });

        // Remove points that are off screen
        this.dataPoints = this.dataPoints.filter(point => point.x > -10);

        // Limit number of points
        if (this.dataPoints.length > this.maxPoints) {
            this.dataPoints.shift();
        }

        this.time++;
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Update Metrics
function updateMetrics() {
    const cadence = document.getElementById('cadence');
    const velocity = document.getElementById('velocity');
    const symmetry = document.getElementById('symmetry');

    if (!cadence) return;

    setInterval(() => {
        // Simulate real-time data updates
        const cadenceValue = (110 + Math.random() * 10).toFixed(0);
        const velocityValue = (1.2 + Math.random() * 0.3).toFixed(2);
        const symmetryValue = (95 + Math.random() * 4).toFixed(1);

        cadence.textContent = cadenceValue;
        velocity.textContent = velocityValue;
        symmetry.textContent = symmetryValue;
    }, 2000);

    // Set initial values
    cadence.textContent = '115';
    velocity.textContent = '1.35';
    symmetry.textContent = '97.2';
}

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add parallax effect to floating elements
function initParallax() {
    const floatingElements = document.querySelectorAll('.floating-element');
    
    document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;
        
        floatingElements.forEach((element, index) => {
            const speed = (index + 1) * 10;
            const x = (mouseX - 0.5) * speed;
            const y = (mouseY - 0.5) * speed;
            
            element.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
        });
    });
}

// Feature cards hover effect
function initFeatureCards() {
    const cards = document.querySelectorAll('.feature-card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.style.setProperty('--mouse-x', `${x}px`);
            this.style.setProperty('--mouse-y', `${y}px`);
        });
    });
}

// Eye tracking for logo pupil
function initEyeTracking() {
    const pupil = document.getElementById('logo-pupil');
    const logoIcon = document.querySelector('.logo-icon');
    
    if (!pupil || !logoIcon) return;
    
    // Constants for the eye
    const eyeCenterX = 25; // Center of eye in SVG viewBox
    const eyeCenterY = 20;
    const pupilRadius = 7;
    const maxDistance = (14 - pupilRadius) * 0.65; // Movement range
    
    // Ensure pupil starts centered
    pupil.setAttribute('cx', eyeCenterX);
    pupil.setAttribute('cy', eyeCenterY);
    
    // Add smooth transition to pupil
    pupil.style.transition = 'cx 0.3s ease-in-out, cy 0.3s ease-in-out';
    
    // Initialize mouse position to the logo's position (so eye stays centered on first track)
    const rect = logoIcon.getBoundingClientRect();
    let currentMouseX = rect.left + rect.width / 2;
    let currentMouseY = rect.top + rect.height / 2;
    let isLooking = false;
    
    // Track mouse position
    document.addEventListener('mousemove', (e) => {
        currentMouseX = e.clientX;
        currentMouseY = e.clientY;
    });
    
    // Function to update pupil position based on cursor
    function updatePupilPosition() {
        const rect = logoIcon.getBoundingClientRect();
        const iconCenterX = rect.left + rect.width / 2;
        const iconCenterY = rect.top + rect.height / 2;
        
        // Calculate angle from eye center to cursor
        const deltaX = currentMouseX - iconCenterX;
        const deltaY = currentMouseY - iconCenterY;
        const angle = Math.atan2(deltaY, deltaX);
        
        // Calculate distance with smooth falloff
        const rawDistance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const normalizedDistance = Math.min(rawDistance / 200, 1) * maxDistance;
        
        // Calculate new pupil position (looking at cursor)
        const newX = eyeCenterX + Math.cos(angle) * normalizedDistance;
        const newY = eyeCenterY + Math.sin(angle) * normalizedDistance;
        
        // Update pupil position
        pupil.setAttribute('cx', newX);
        pupil.setAttribute('cy', newY);
    }
    
    // Function to look at cursor
    function glanceAtCursor() {
        if (isLooking) return;
        
        isLooking = true;
        
        // Temporarily disable transition for smoother tracking
        pupil.style.transition = 'cx 0.1s ease-out, cy 0.1s ease-out';
        
        // Track cursor continuously for 5 seconds
        const trackingInterval = setInterval(updatePupilPosition, 16); // ~60fps
        
        // Return to center after 5 seconds, then wait before next glance
        setTimeout(() => {
            clearInterval(trackingInterval);
            pupil.style.transition = 'cx 0.3s ease-in-out, cy 0.3s ease-in-out';
            pupil.setAttribute('cx', eyeCenterX);
            pupil.setAttribute('cy', eyeCenterY);
            isLooking = false;
            
            // Wait 5 seconds before scheduling the next glance
            scheduleNextGlance();
        }, 5000); // Track for 5 seconds
    }
    
    // Schedule the next glance after a 10 second break
    function scheduleNextGlance() {
        setTimeout(() => {
            glanceAtCursor();
        }, 7000); // 10 second break between glances
    }
    
    // Start centered, then begin the glancing cycle after a brief moment
    setTimeout(() => {
        glanceAtCursor();
    }, 500); // Small delay to show centered state on load
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initStatCounters();
    new GaitVisualization('gaitCanvas');
    updateMetrics();
    initParallax();
    initFeatureCards();
    initEyeTracking();
});

// Add button click handlers
document.querySelectorAll('.btn-primary, .btn-secondary, .nav-cta').forEach(button => {
    button.addEventListener('click', function() {
        // Add ripple effect
        const ripple = document.createElement('span');
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.5);
            width: 100px;
            height: 100px;
            margin-left: -50px;
            margin-top: -50px;
            animation: ripple 0.6s;
            pointer-events: none;
        `;
        
        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        from {
            opacity: 1;
            transform: scale(0);
        }
        to {
            opacity: 0;
            transform: scale(2);
        }
    }
`;
document.head.appendChild(style);

