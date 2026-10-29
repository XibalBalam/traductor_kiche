/**
 * Ch'aweb'al - Sistema Clínico K'iche' ↔ Español
 * Pull-to-Refresh & Hold Gesture Controller para WebApp Móvil (PWA Standalone)
 * 
 * Permite que en cualquier pantalla de la aplicación móvil (especialmente en modo PWA
 * standalone donde no hay barra de navegación ni botón de recarga), al deslizar
 * suficientemente hacia abajo desde el tope superior y sostener un instante, se
 * ejecute una recarga limpia y segura de la página.
 */

(function () {
    'use strict';

    // Evitar inicialización duplicada
    if (window.__chawabalPullToRefreshInitialized) return;
    window.__chawabalPullToRefreshInitialized = true;

    // Configuración calibrada para una experiencia móvil natural y no intrusiva
    const CONFIG = {
        PULL_MIN_START: 5,        // Pixeles mínimos de scroll para considerar tope superior
        TRIGGER_RAW_DELTA: 120,    // Desplazamiento del dedo necesario para alcanzar el umbral (~120px)
        MAX_VISUAL_PULL: 95,      // Desplazamiento visual máximo del indicador (resistencia elástica)
        HOLD_DURATION_MS: 450,    // Tiempo de sostenimiento necesario al llegar al umbral (ms)
        CIRCUMFERENCE: 69.12,     // 2 * PI * 11 (radio del anillo SVG)
        RESISTANCE_FACTOR: 0.42   // Amortiguación física del arrastre
    };

    let startY = 0;
    let startX = 0;
    let currentY = 0;
    let isTracking = false;
    let isPulling = false;
    let isHolding = false;
    let isReloading = false;
    let holdTimer = null;
    let holdStartTime = 0;
    let currentPull = 0;
    let animFrameId = null;

    // Inyectar estilos CSS aislados para el indicador
    function injectStyles() {
        if (document.getElementById('chawabal-ptr-styles')) return;

        const style = document.createElement('style');
        style.id = 'chawabal-ptr-styles';
        style.textContent = `
            .chawabal-ptr-container {
                position: fixed;
                top: max(12px, env(safe-area-inset-top, 12px));
                left: 50%;
                transform: translate(-50%, -120px);
                z-index: 999999;
                pointer-events: none;
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 9px 18px;
                border-radius: 9999px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 12.5px;
                font-weight: 600;
                letter-spacing: -0.01em;
                user-select: none;
                -webkit-user-select: none;
                opacity: 0;
                transition: transform 0.12s cubic-bezier(0.2, 0.8, 0.4, 1), opacity 0.15s ease, background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
                will-change: transform, opacity;
                /* Modo Oscuro por defecto (Ch'aweb'al dark) */
                background: rgba(15, 23, 42, 0.94);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                color: #F8FAFC;
                border: 1px solid rgba(16, 185, 129, 0.35);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65), 0 0 15px rgba(16, 185, 129, 0.2);
            }

            /* Adaptación al Modo Claro cuando la clase .light-mode está activa */
            :root.light-mode .chawabal-ptr-container,
            body.light-mode .chawabal-ptr-container {
                background: rgba(255, 255, 255, 0.96);
                color: #102A25;
                border: 1px solid rgba(16, 185, 129, 0.45);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18), 0 0 15px rgba(16, 185, 129, 0.25);
            }

            .chawabal-ptr-container.active-hold {
                border-color: #10B981 !important;
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7), 0 0 24px rgba(16, 185, 129, 0.5) !important;
                transform: translate(-50%, var(--ptr-y, 65px)) scale(1.03) !important;
            }

            .chawabal-ptr-icon-box {
                position: relative;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }

            .chawabal-ptr-ring-svg {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                transform: rotate(-90deg);
            }

            .chawabal-ptr-ring-bg {
                fill: none;
                stroke: rgba(255, 255, 255, 0.15);
                stroke-width: 2.4;
            }
            :root.light-mode .chawabal-ptr-ring-bg,
            body.light-mode .chawabal-ptr-ring-bg {
                stroke: rgba(16, 42, 37, 0.15);
            }

            .chawabal-ptr-ring-fg {
                fill: none;
                stroke: #10B981;
                stroke-width: 2.4;
                stroke-linecap: round;
                stroke-dasharray: 69.12;
                stroke-dashoffset: 69.12;
                transition: stroke-dashoffset 0.08s linear, stroke 0.2s ease;
            }

            .chawabal-ptr-arrow {
                width: 13px;
                height: 13px;
                fill: currentColor;
                transition: transform 0.12s cubic-bezier(0.2, 0.8, 0.4, 1);
            }

            .chawabal-ptr-text {
                white-space: nowrap;
                transition: color 0.2s ease;
            }

            /* Animación de giro constante cuando está recargando */
            @keyframes chawabalPtrSpin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .chawabal-ptr-spinning {
                animation: chawabalPtrSpin 0.75s linear infinite !important;
            }
        `;
        document.head.appendChild(style);
    }

    // Inyectar el elemento DOM del indicador
    let ptrElement = null;
    let ptrRing = null;
    let ptrArrow = null;
    let ptrText = null;

    function createDOM() {
        if (ptrElement) return;

        ptrElement = document.createElement('div');
        ptrElement.id = 'chawabal-pull-refresh';
        ptrElement.className = 'chawabal-ptr-container';
        ptrElement.setAttribute('aria-hidden', 'true');

        ptrElement.innerHTML = `
            <div class="chawabal-ptr-icon-box">
                <svg class="chawabal-ptr-ring-svg" viewBox="0 0 28 28">
                    <circle class="chawabal-ptr-ring-bg" cx="14" cy="14" r="11"></circle>
                    <circle class="chawabal-ptr-ring-fg" id="chawabal-ptr-ring-fg" cx="14" cy="14" r="11"></circle>
                </svg>
                <svg class="chawabal-ptr-arrow" id="chawabal-ptr-arrow" viewBox="0 0 24 24">
                    <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                </svg>
            </div>
            <span class="chawabal-ptr-text" id="chawabal-ptr-text">Desliza para recargar</span>
        `;

        document.body.appendChild(ptrElement);

        ptrRing = document.getElementById('chawabal-ptr-ring-fg');
        ptrArrow = document.getElementById('chawabal-ptr-arrow');
        ptrText = document.getElementById('chawabal-ptr-text');
    }

    // Obtener la posición de scroll actual de la ventana
    function getScrollTop() {
        return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
    }

    // Verificar si el toque empezó en un elemento interactivo que no debe interferir
    function isInteractiveTarget(target) {
        if (!target) return false;
        // Si el usuario toca el micrófono, un botón, un campo de texto, un slider o una onda de audio
        const interactive = target.closest(
            '#micBtn, .btn-mic, button, input, textarea, select, [contenteditable], canvas, a, .wavesurfer-handle, [role="button"], [role="slider"]'
        );
        return Boolean(interactive);
    }

    // Actualizar la interfaz visual del indicador
    function updateUI(rawDeltaY, progress) {
        if (!ptrElement) return;

        // Limitar desplazamiento con amortiguación
        const visualPull = Math.min(CONFIG.MAX_VISUAL_PULL, rawDeltaY * CONFIG.RESISTANCE_FACTOR);
        currentPull = visualPull;

        // Posición Y
        const translateY = visualPull - 65; // Desciende desde -65px hasta ~30px bajo el margen superior
        ptrElement.style.setProperty('--ptr-y', `${visualPull}px`);
        ptrElement.style.transform = `translate(-50%, ${translateY}px)`;
        ptrElement.style.opacity = Math.min(1, progress * 1.35).toString();

        // Rotación del icono de flecha
        if (ptrArrow && !isReloading) {
            ptrArrow.style.transform = `rotate(${Math.min(360, progress * 360)}deg)`;
        }

        // Anillo de progreso SVG (stroke-dashoffset)
        if (ptrRing && !isHolding && !isReloading) {
            const offset = CONFIG.CIRCUMFERENCE * (1 - Math.min(0.75, progress * 0.75));
            ptrRing.style.strokeDashoffset = offset.toString();
        }

        // Texto informativo
        if (ptrText && !isHolding && !isReloading) {
            if (rawDeltaY >= CONFIG.TRIGGER_RAW_DELTA) {
                ptrText.textContent = "¡Sostén para recargar!";
            } else {
                ptrText.textContent = "Desliza para recargar";
            }
        }
    }

    // Iniciar animación suave para completar el anillo durante el sostenimiento
    function startHoldProgress() {
        if (isHolding) return;
        isHolding = true;
        holdStartTime = Date.now();

        if (ptrElement) {
            ptrElement.classList.add('active-hold');
        }
        if (ptrText) {
            ptrText.textContent = "¡Sostén un momento...!";
        }

        // Vibración háptica suave si está disponible en el móvil
        if (navigator.vibrate) {
            try { navigator.vibrate(35); } catch (e) { }
        }

        function stepRing() {
            if (!isHolding || isReloading) return;
            const elapsed = Date.now() - holdStartTime;
            const factor = Math.min(1, elapsed / CONFIG.HOLD_DURATION_MS);

            // Completar del 75% al 100%
            if (ptrRing) {
                const currentOffset = CONFIG.CIRCUMFERENCE * (0.25 * (1 - factor));
                ptrRing.style.strokeDashoffset = currentOffset.toString();
            }

            if (factor < 1) {
                animFrameId = requestAnimationFrame(stepRing);
            }
        }
        animFrameId = requestAnimationFrame(stepRing);

        // Disparar recarga al completar el tiempo sostenido
        holdTimer = setTimeout(() => {
            triggerReload();
        }, CONFIG.HOLD_DURATION_MS);
    }

    // Cancelar el estado de sostenimiento si el usuario regresa hacia arriba
    function cancelHoldProgress() {
        if (!isHolding) return;
        isHolding = false;
        if (holdTimer) {
            clearTimeout(holdTimer);
            holdTimer = null;
        }
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
        if (ptrElement) {
            ptrElement.classList.remove('active-hold');
        }
    }

    // Ejecutar la recarga de la página con feedback visual total
    function triggerReload() {
        if (isReloading) return;
        isReloading = true;
        cancelHoldProgress();

        // Vibración de confirmación
        if (navigator.vibrate) {
            try { navigator.vibrate([25, 40, 30]); } catch (e) { }
        }

        // Estado visual de recargando
        if (ptrElement) {
            ptrElement.classList.add('active-hold');
            ptrElement.style.transform = `translate(-50%, 25px)`;
            ptrElement.style.opacity = '1';
        }
        if (ptrRing) {
            ptrRing.style.strokeDashoffset = '0';
        }
        if (ptrArrow) {
            ptrArrow.classList.add('chawabal-ptr-spinning');
        }
        if (ptrText) {
            ptrText.textContent = "Recargando...";
        }

        // Pequeña pausa de 200ms para que el usuario aprecie el estado visual y luego recargar
        setTimeout(() => {
            try {
                // 1. Intento de recarga estándar
                window.location.reload();
            } catch (e) {
                // 2. Fallback con cache-bust si el navegador estuviera bloqueado por PWA
                window.location.href = window.location.pathname + (window.location.search ? window.location.search + '&' : '?') + 'r=' + Date.now();
            }

            // Fallback de seguridad adicional si después de 1 segundo sigue sin responder
            setTimeout(() => {
                window.location.href = window.location.href;
            }, 1000);
        }, 220);
    }

    // Resetear el indicador con animación elástica de retirada
    function resetUI() {
        cancelHoldProgress();
        isTracking = false;
        isPulling = false;

        if (ptrElement && !isReloading) {
            ptrElement.style.transition = 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.25), opacity 0.25s ease';
            ptrElement.style.transform = 'translate(-50%, -120px)';
            ptrElement.style.opacity = '0';

            setTimeout(() => {
                if (!isPulling && !isReloading && ptrElement) {
                    ptrElement.style.transition = '';
                    if (ptrRing) ptrRing.style.strokeDashoffset = CONFIG.CIRCUMFERENCE.toString();
                    if (ptrArrow) {
                        ptrArrow.style.transform = 'rotate(0deg)';
                        ptrArrow.classList.remove('chawabal-ptr-spinning');
                    }
                    if (ptrText) ptrText.textContent = "Desliza para recargar";
                }
            }, 320);
        }
    }

    // MANEJADORES DE EVENTOS TOUCH
    function onTouchStart(e) {
        if (isReloading) return;
        if (!e.touches || e.touches.length !== 1) return;

        // Solo permitir si estamos en el tope de la pantalla
        if (getScrollTop() > CONFIG.PULL_MIN_START) return;

        // No interferir si el usuario interactúa con botones, micrófono, etc.
        if (isInteractiveTarget(e.target)) return;

        startY = e.touches[0].clientY;
        startX = e.touches[0].clientX;
        currentY = startY;
        isTracking = true;
        isPulling = false;
        cancelHoldProgress();
    }

    function onTouchMove(e) {
        if (!isTracking || isReloading) return;
        if (!e.touches || e.touches.length !== 1) return;

        currentY = e.touches[0].clientY;
        const deltaY = currentY - startY;
        const deltaX = e.touches[0].clientX - startX;

        // Si el usuario se desplazó hacia arriba o el scroll ya no está en el tope
        if (deltaY <= 0 || getScrollTop() > CONFIG.PULL_MIN_START) {
            if (isPulling) resetUI();
            return;
        }

        // Si el movimiento es predominantemente horizontal (ej. slider o swipe lateral), ignorar
        if (Math.abs(deltaX) > deltaY * 0.9 && !isPulling) {
            isTracking = false;
            return;
        }

        // Iniciar arrastre si supera el umbral de detección inicial
        if (deltaY > 15) {
            isPulling = true;
            createDOM();

            const progress = Math.min(1.2, deltaY / CONFIG.TRIGGER_RAW_DELTA);
            updateUI(deltaY, progress);

            // Si alcanzó el umbral sustancial ("deslice demasiado hacia abajo")
            if (deltaY >= CONFIG.TRIGGER_RAW_DELTA) {
                startHoldProgress();
            } else {
                cancelHoldProgress();
            }

            // Prevenir comportamiento de rebote nativo conflictivo cuando se arrastra activamente
            if (e.cancelable && deltaY > 30) {
                e.preventDefault();
            }
        }
    }

    function onTouchEnd(e) {
        if (!isTracking || isReloading) return;

        const deltaY = currentY - startY;

        // Si el usuario sostuvo el tiempo completo, el timer ya habrá disparado triggerReload().
        // Si el usuario alcanzó el umbral completo y soltó después de al menos 200ms de sostenimiento:
        if (isPulling && deltaY >= CONFIG.TRIGGER_RAW_DELTA && (Date.now() - holdStartTime >= 180)) {
            triggerReload();
            return;
        }

        // Si no completó o canceló
        resetUI();
    }

    function onTouchCancel() {
        if (!isReloading) {
            resetUI();
        }
    }

    // Inicializar cuando el DOM esté listo
    function init() {
        injectStyles();
        createDOM();

        // Registrar escuchadores de eventos pasivos/no pasivos según convenga
        window.addEventListener('touchstart', onTouchStart, { passive: true });
        window.addEventListener('touchmove', onTouchMove, { passive: false });
        window.addEventListener('touchend', onTouchEnd, { passive: true });
        window.addEventListener('touchcancel', onTouchCancel, { passive: true });

        console.log("✅ Ch'aweb'al Mobile Pull-to-Refresh inicializado exitosamente.");
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
