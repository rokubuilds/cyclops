// People Class
class People {
    constructor(name, image) {
        this.name = name;
        this.image = image;
    }
}

// Create instances for each person
const caroline = new People('caroline', 'images/caroline_walk.png');
const henrik = new People('henrik', 'images/henrik_walk.png');
const mofei = new People('mofei', 'images/mofei_walk.png');
const rohan = new People('rohan', 'images/rohan_walk.png');

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
    const avgSpeed = document.getElementById('avg-speed');
    const avgHeight = document.getElementById('avg-height');
    const stepRhythm = document.getElementById('step-rhythm');

    if (!avgSpeed) return;

    setInterval(() => {
        // Simulate real-time data updates
        const avgSpeedValue = (1.2 + Math.random() * 0.3).toFixed(2);
        const avgHeightValue = (1.65 + Math.random() * 0.05).toFixed(2);
        const stepRhythmValue = (1.8 + Math.random() * 0.2).toFixed(1);

        avgSpeed.textContent = avgSpeedValue;
        avgHeight.textContent = avgHeightValue;
        stepRhythm.textContent = stepRhythmValue;
    }, 2000);

    // Set initial values
    avgSpeed.textContent = '1.35';
    avgHeight.textContent = '1.68';
    stepRhythm.textContent = '1.9';
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

// Trigger Button Functionality
function initTriggerButton() {
    const triggerBtn = document.getElementById('trigger-btn');
    const portraitImage = document.getElementById('portrait-image');
    const portraitPlaceholder = document.getElementById('portrait-placeholder');
    const authStatus = document.getElementById('auth-status');
    const authMessage = document.getElementById('auth-message');
    const rejectX = document.getElementById('reject-x');
    const checkmark = document.getElementById('auth-checkmark');
    const rejectIcon = document.getElementById('auth-reject-x');
    
    if (!triggerBtn || !portraitImage) return;
    
    // Array of all people instances
    const peopleArray = [caroline, henrik, mofei, rohan];
    
    triggerBtn.addEventListener('click', () => {
        // Hide reject X if visible
        if (rejectX) rejectX.style.display = 'none';
        
        // Randomly select one person
        const randomPerson = peopleArray[Math.floor(Math.random() * peopleArray.length)];
        
        // Update the image
        portraitImage.src = randomPerson.image;
        portraitImage.style.display = 'block';
        
        // Hide placeholder text
        if (portraitPlaceholder) {
            portraitPlaceholder.style.display = 'none';
        }
        
        // Add a fade-in effect
        portraitImage.style.opacity = '0';
        setTimeout(() => {
            portraitImage.style.transition = 'opacity 0.5s ease-in-out';
            portraitImage.style.opacity = '1';
        }, 10);
        
        // Show authorization message
        if (authStatus && authMessage) {
            const capitalizedName = randomPerson.name.charAt(0).toUpperCase() + randomPerson.name.slice(1);
            authMessage.textContent = `${capitalizedName} is authorized!`;
            authMessage.style.color = '#10b981';
            authStatus.style.display = 'flex';
            authStatus.style.borderColor = '#10b981';
            
            // Show checkmark, hide reject X
            if (checkmark) checkmark.style.display = 'block';
            if (rejectIcon) rejectIcon.style.display = 'none';
            
            // Flash the authorization box green
            authStatus.style.background = 'rgba(16, 185, 129, 0.3)';
            setTimeout(() => {
                authStatus.style.transition = 'background 0.5s ease';
                authStatus.style.background = 'var(--bg-secondary)';
            }, 200);
            
            // Animate the checkmark
            if (checkmark) {
                checkmark.style.transform = 'scale(0)';
                setTimeout(() => {
                    checkmark.style.transition = 'transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                    checkmark.style.transform = 'scale(1)';
                }, 50);
            }
        }
        
        console.log(`Loaded: ${randomPerson.name} - AUTHORIZED`);
    });
}

// Reject Button Functionality
function initRejectButton() {
    const rejectBtn = document.getElementById('reject-btn');
    const portraitImage = document.getElementById('portrait-image');
    const portraitPlaceholder = document.getElementById('portrait-placeholder');
    const authStatus = document.getElementById('auth-status');
    const authMessage = document.getElementById('auth-message');
    const rejectX = document.getElementById('reject-x');
    const checkmark = document.getElementById('auth-checkmark');
    const rejectIcon = document.getElementById('auth-reject-x');
    
    if (!rejectBtn) return;
    
    rejectBtn.addEventListener('click', () => {
        // Hide any loaded image
        if (portraitImage) {
            portraitImage.style.display = 'none';
        }
        
        // Show placeholder text
        if (portraitPlaceholder) {
            portraitPlaceholder.style.display = 'block';
        }
        
        // Show big red X overlay
        if (rejectX) {
            rejectX.style.display = 'flex';
            rejectX.style.opacity = '0';
            setTimeout(() => {
                rejectX.style.transition = 'opacity 0.3s ease-in-out';
                rejectX.style.opacity = '1';
            }, 10);
        }
        
        // Show rejection message
        if (authStatus && authMessage) {
            authMessage.textContent = 'UNAUTHORIZED USER REJECTED!';
            authMessage.style.color = '#ef4444';
            authStatus.style.display = 'flex';
            authStatus.style.borderColor = '#ef4444';
            
            // Show reject X icon, hide checkmark
            if (checkmark) checkmark.style.display = 'none';
            if (rejectIcon) rejectIcon.style.display = 'block';
            
            // Flash the authorization box red
            authStatus.style.background = 'rgba(239, 68, 68, 0.3)';
            setTimeout(() => {
                authStatus.style.transition = 'background 0.5s ease';
                authStatus.style.background = 'var(--bg-secondary)';
            }, 200);
            
            // Animate the reject icon
            if (rejectIcon) {
                rejectIcon.style.transform = 'scale(0) rotate(0deg)';
                setTimeout(() => {
                    rejectIcon.style.transition = 'transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                    rejectIcon.style.transform = 'scale(1) rotate(90deg)';
                }, 50);
            }
        }
        
        console.log('UNAUTHORIZED USER - REJECTED');
    });
}

// Calculate nonchalance score based on gait features
function calculateNonchalance() {
    // Get all the feature values from the page
    const features = {
        avg_speed: parseFloat(document.getElementById('demo-avg-speed').textContent),
        speed_std: parseFloat(document.getElementById('demo-speed-std').textContent),
        avg_accel: parseFloat(document.getElementById('demo-avg-accel').textContent),
        accel_std: parseFloat(document.getElementById('demo-accel-std').textContent),
        step_rhythm: parseFloat(document.getElementById('demo-step-rhythm').textContent),
        lateral_movement: parseFloat(document.getElementById('demo-lateral').textContent),
        vertical_range: parseFloat(document.getElementById('demo-vertical-range').textContent),
        vertical_velocity_std: parseFloat(document.getElementById('demo-vert-velocity-std').textContent)
    };
    
    // 1. Speed component (30% weight)
    const speed_target = 1.2; // m/s
    const speed_deviation = Math.abs(features.avg_speed - speed_target) / speed_target;
    const speed_score = Math.max(0, 1 - speed_deviation) * 100;
    
    const speed_consistency = features.avg_speed > 0 
        ? Math.max(0, 1 - (features.speed_std / features.avg_speed)) * 100 
        : 0;
    const speed_component = (speed_score * 0.6 + speed_consistency * 0.4) * 0.30;
    
    // 2. Rhythm component (25% weight)
    const rhythm_irregularity = Math.min(features.step_rhythm / 0.5, 1.0) * 100;
    const rhythm_component = rhythm_irregularity * 0.25;
    
    // 3. Smoothness component (25% weight)
    const accel_smoothness = Math.max(0, 1 - (features.accel_std / (features.avg_accel + 0.01))) * 100;
    const vertical_smoothness = Math.max(0, 1 - (features.vertical_velocity_std / (features.avg_speed + 0.01))) * 100;
    const smoothness_component = (accel_smoothness * 0.6 + vertical_smoothness * 0.4) * 0.25;
    
    // 4. Posture component (20% weight)
    const lateral_target = 0.15;
    const lateral_deviation = Math.abs(features.lateral_movement - lateral_target);
    const lateral_score = Math.max(0, 1 - (lateral_deviation / lateral_target)) * 100;
    
    const vertical_bounce_penalty = Math.min(features.vertical_range / 0.3, 1.0) * 100;
    const posture_score = lateral_score * 0.6 + (100 - vertical_bounce_penalty) * 0.4;
    const posture_component = posture_score * 0.20;
    
    // Combine all components
    const nonchalance_score = speed_component + rhythm_component + smoothness_component + posture_component;
    
    return Math.round(nonchalance_score);
}

// Calculate BAC (Blood Alcohol Content) - returns realistic values 0.00% to ~0.20%
function estimateBAC() {
    const features = {
        avg_speed: parseFloat(document.getElementById('demo-avg-speed').textContent),
        speed_std: parseFloat(document.getElementById('demo-speed-std').textContent),
        lateral_movement: parseFloat(document.getElementById('demo-lateral').textContent),
        vertical_velocity_std: parseFloat(document.getElementById('demo-vert-velocity-std').textContent),
        accel_std: parseFloat(document.getElementById('demo-accel-std').textContent),
        step_rhythm: parseFloat(document.getElementById('demo-step-rhythm').textContent),
        height_std: parseFloat(document.getElementById('demo-height-std').textContent),
        vertical_range: parseFloat(document.getElementById('demo-vertical-range').textContent)
    };
    
    // Baseline "sober" values
    const baseline = {
        avg_speed: 1.35,
        lateral_movement: 0.06,
        vertical_velocity_std: 0.11,
        speed_std: 0.18,
        accel_std: 0.15,
        step_rhythm: 1.9,
        height_std: 0.04,
        vertical_range: 0.08
    };
    
    let impairment_score = 0;
    
    // 1. Excessive lateral sway (35% weight) - drunk people sway more
    const sway_excess = Math.max(0, (features.lateral_movement - baseline.lateral_movement) / baseline.lateral_movement);
    impairment_score += Math.min(sway_excess * 2, 1.0) * 0.35;
    
    // 2. Speed inconsistency (25% weight) - drunk walking is erratic
    const speed_variance = Math.max(0, (features.speed_std - baseline.speed_std) / baseline.speed_std);
    impairment_score += Math.min(speed_variance * 1.5, 1.0) * 0.25;
    
    // 3. Acceleration jerkiness (20% weight) - lack of motor control
    const accel_excess = Math.max(0, (features.accel_std - baseline.accel_std) / baseline.accel_std);
    impairment_score += Math.min(accel_excess * 1.5, 1.0) * 0.20;
    
    // 4. Vertical instability (15% weight) - bobbing up and down
    const vertical_excess = Math.max(0, (features.vertical_velocity_std - baseline.vertical_velocity_std) / baseline.vertical_velocity_std);
    impairment_score += Math.min(vertical_excess * 1.2, 1.0) * 0.15;
    
    // 5. Overall speed reduction (5% weight) - intoxicated people often walk slower
    if (features.avg_speed < baseline.avg_speed * 0.9) {
        const speed_reduction = (baseline.avg_speed - features.avg_speed) / baseline.avg_speed;
        impairment_score += Math.min(speed_reduction * 2, 1.0) * 0.05;
    }
    
    // Convert impairment_score (0-1) to realistic BAC (0.00% to 0.20%)
    // 0.08% is legal limit, 0.15% is very drunk, 0.20% is dangerously high
    const bac = impairment_score * 0.20;
    
    // Return as percentage string with 2 decimal places (e.g., "0.05")
    return bac.toFixed(2);
}

// Calculate and BAC Button Progress Functionality
function initCalculateButtons() {
    const calculateBtn = document.getElementById('calculate-nonchalance-btn');
    const bacBtn = document.getElementById('estimate-bac-btn');
    
    function setupProgressButton(button) {
        if (!button) return;
        
        const progressBar = button.querySelector('.progress-bar-bg');
        const textSpan = button.querySelector('span');
        
        if (!progressBar || !textSpan) return;
        
        const originalText = textSpan.textContent;
        
        button.addEventListener('click', (e) => {
            // Prevent if already running
            if (button.classList.contains('calculating')) return;
            
            button.classList.add('calculating');
            button.style.cursor = 'wait';
            
            // Make button purple (secondary color) instead of transparent
            button.style.background = '#7c3aed';
            button.style.border = '1px solid var(--primary)';
            
            // Determine the loading text based on button
            const loadingText = originalText.includes('Nonchalance') ? 'Calculating' : 'Estimating';
            
            // Animate dots
            let dotCount = 0;
            textSpan.textContent = loadingText;
            
            const dotInterval = setInterval(() => {
                dotCount = (dotCount + 1) % 4;
                textSpan.textContent = loadingText + '.'.repeat(dotCount);
            }, 500);
            
            // Reset progress bar
            progressBar.style.transition = 'none';
            progressBar.style.width = '0%';
            
            // Force reflow to restart animation
            progressBar.offsetHeight;
            
            // Start progress animation (4 seconds for longer delay)
            progressBar.style.transition = 'width 4s linear';
            progressBar.style.width = '100%';
            
            // After 4 seconds, show calculated number
            setTimeout(() => {
                // Stop dot animation
                clearInterval(dotInterval);
                
                // Calculate based on button type
                let displayText;
                if (originalText.includes('Nonchalance')) {
                    const calculatedNum = calculateNonchalance();
                    displayText = calculatedNum + '%';
                } else {
                    // Calculate BAC (returns value like "0.05")
                    const bacValue = estimateBAC();
                    displayText = bacValue + '%';
                }
                
                // Make number bigger and bolder
                textSpan.style.fontSize = '2rem';
                textSpan.style.fontWeight = '900';
                textSpan.textContent = displayText;
                
                // Keep the cyan background (don't hide progress bar)
                button.style.background = 'var(--primary)';
                button.style.border = 'none';
                
                // After 6 seconds showing the number (longer), reset
                setTimeout(() => {
                    textSpan.textContent = originalText;
                    textSpan.style.fontSize = '';
                    textSpan.style.fontWeight = '';
                    button.classList.remove('calculating');
                    button.style.cursor = 'pointer';
                    
                    // Reset progress bar
                    progressBar.style.width = '0%';
                    progressBar.style.transition = 'none';
                }, 6000);
            }, 4000);
        });
    }
    
    setupProgressButton(calculateBtn);
    setupProgressButton(bacBtn);
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initStatCounters();
    new GaitVisualization('gaitCanvas');
    updateMetrics();
    initParallax();
    initFeatureCards();
    initEyeTracking();
    initTriggerButton();
    initRejectButton();
    initCalculateButtons();
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

