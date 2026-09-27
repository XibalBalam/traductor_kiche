const fakeCursor = document.createElement('div');
fakeCursor.id = 'fake-cursor';
fakeCursor.innerHTML = '<span class="material-symbols-outlined" style="font-size: 28px; color: #fff; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5)); transform: rotate(-15deg);">mouse</span>';
fakeCursor.style.position = 'fixed';
fakeCursor.style.pointerEvents = 'none';
fakeCursor.style.zIndex = '1000000';
fakeCursor.style.transition = 'all 0.8s cubic-bezier(0.25, 1, 0.5, 1)';
fakeCursor.style.opacity = '0';
fakeCursor.style.top = '50%';
fakeCursor.style.left = '50%';
document.body.appendChild(fakeCursor);

function animateCursor(selector, actionCallback, delayBeforeAction = 800) {
    const el = document.querySelector(selector);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    
    fakeCursor.style.opacity = '1';
    fakeCursor.style.top = (rect.top + rect.height / 2) + 'px';
    fakeCursor.style.left = (rect.left + rect.width / 2) + 'px';
    
    setTimeout(() => {
        fakeCursor.style.transform = 'scale(0.8) rotate(-15deg)';
        setTimeout(() => {
            fakeCursor.style.transform = 'scale(1) rotate(-15deg)';
            if (actionCallback) actionCallback();
        }, 150);
    }, delayBeforeAction);
}

function hideCursor() {
    fakeCursor.style.opacity = '0';
}

function resetTourUI() {
    if (typeof setMode === 'function') setMode('kiche-es');
    document.getElementById('resultBox').classList.add('hidden');
    document.getElementById('resultBox').classList.remove('flex', 'opacity-100');
    document.getElementById('anatomyContainer').style.maxHeight = '0px';
    document.getElementById('anatomyContainer').style.opacity = '0';
    document.getElementById('transcriptionInput').value = '';
    document.getElementById('translationText').textContent = '';
    document.getElementById('status').textContent = "Mantén presionado para hablar en K'iche'";
    
    const fab = document.getElementById('summaryFabContainer');
    if (fab) {
        fab.classList.add('translate-y-24', 'opacity-0', 'pointer-events-none');
        fab.classList.remove('translate-y-0', 'opacity-100', 'pointer-events-auto');
    }
    
    const modal = document.getElementById('summaryModal');
    const content = document.getElementById('summaryModalContent');
    if (modal && content) {
        modal.classList.add('opacity-0', 'pointer-events-none');
        content.classList.add('scale-95');
        content.classList.remove('scale-100');
        document.getElementById('summaryTextarea').value = "";
        document.getElementById('btnGenerateSummary').textContent = "Generar Ahora";
    }
}

function startTour() {
    const driver = window.driver.js.driver;
    const hasAdminControls = document.querySelector('a[href="/train"]') || document.querySelector('a[href="/admin"]');
    
    resetTourUI();
    let audioTour = new Audio();
    
    function playTTS(url, text) {
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })
        .then(res => res.blob())
        .then(blob => {
            audioTour.src = URL.createObjectURL(blob);
            audioTour.play().catch(e => console.log('Audio autoplay prevented'));
        })
        .catch(err => console.error("Error playing tour audio:", err));
    }
    
    let tourTimeouts = [];
    function clearTourTimeouts() {
        tourTimeouts.forEach(clearTimeout);
        tourTimeouts.forEach(clearInterval);
        tourTimeouts = [];
    }
    function registerTimeout(fn, delay) {
        const id = setTimeout(fn, delay);
        tourTimeouts.push(id);
        return id;
    }
    function registerInterval(fn, delay) {
        const id = setInterval(fn, delay);
        tourTimeouts.push(id);
        return id;
    }

    function refreshSpotlight() {
        if (!window.tourDriver) return;
        const iv = registerInterval(() => window.tourDriver.refresh(), 30);
        registerTimeout(() => clearInterval(iv), 800);
    }

    const steps = [
        {
            element: '#modeToggle',
            popover: {
                title: 'Traducción Bilingüe',
                description: 'Cambia entre traducir del K\'iche\' al Español o viceversa. La interfaz se adaptará de inmediato.',
                side: "bottom",
                align: 'start'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">translate</span> Traducción Bilingüe';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
                registerTimeout(() => {
                    animateCursor('#btnEsToKiche', () => {
                        if (typeof setMode === 'function') setMode('es-kiche');
                        registerTimeout(() => {
                            animateCursor('#btnKicheToEs', () => {
                                if (typeof setMode === 'function') setMode('kiche-es');
                                registerTimeout(hideCursor, 500);
                            }, 800);
                        }, 1200);
                    });
                }, 500);
            },
            onDeselected: () => { clearTourTimeouts(); hideCursor(); }
        },
        {
            element: '.container',
            popover: {
                title: 'Dictado Inteligente',
                description: 'Solo mantén presionado el micrófono para hablar.',
                side: "bottom",
                align: 'center'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">mic</span> Dictado Inteligente';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
                
                document.getElementById('dynamicContentWrapper').style.opacity = '0';
                
                registerTimeout(() => {
                    animateCursor('#micBtn', () => {
                        const mic = document.getElementById('micBtn');
                        mic.classList.add('recording');
                        document.getElementById('status').textContent = "Escuchando...";
                        
                        registerTimeout(() => {
                            mic.classList.remove('recording');
                            document.getElementById('status').textContent = "Traduciendo...";
                            hideCursor();
                            
                            registerTimeout(() => {
                                window.tourDriver.moveNext();
                            }, 1000);
                        }, 1500);
                    });
                }, 500);
            },
            onDeselected: () => { clearTourTimeouts(); hideCursor(); const mic = document.getElementById('micBtn'); mic.classList.remove('recording'); document.getElementById('dynamicContentWrapper').style.opacity = '1'; }
        },
        {
            element: '#dynamicContentWrapper',
            popover: {
                title: 'Análisis Anatómico',
                description: 'La Inteligencia Artificial traduce lo que escuchó, detecta si mencionas algún dolor, y te ilumina exactamente la zona afectada en el modelo anatómico. (Asegúrate de tener volumen).',
                side: "bottom",
                align: 'center'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">medical_services</span> Análisis Anatómico';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
                document.getElementById('dynamicContentWrapper').style.opacity = '1';
                
                const ti = document.getElementById('transcriptionInput');
                const tt = document.getElementById('translationText');
                const rBox = document.getElementById('resultBox');
                
                rBox.style.display = '';
                rBox.classList.remove('hidden', 'opacity-0');
                rBox.classList.add('flex', 'opacity-100');
                ti.value = "K'ax nu pam";
                document.getElementById('status').textContent = "¡Traducción completada!";
                
                if (typeof analyzeMedicalContext === 'function') {
                    analyzeMedicalContext("k'ax nu pam me duele el estomago");
                }
                
                refreshSpotlight();

                tt.textContent = "";
                let text = "Me duele el estómago.";
                let i = 0;
                let ival = registerInterval(() => {
                    tt.textContent += text.charAt(i);
                    i++;
                    if(i >= text.length) {
                        clearInterval(ival);
                        playTTS('/tts/es', text);
                    }
                }, 40);
            },
            onDeselected: () => { clearTourTimeouts(); audioTour.pause(); }
        },
        {
            element: '#dynamicContentWrapper',
            popover: {
                title: 'Traducción Inversa',
                description: 'Funciona exactamente igual al revés: el doctor habla en Español y el sistema responde en K\'iche\', iluminando la misma zona.',
                side: "bottom",
                align: 'center'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">swap_horiz</span> Traducción Inversa';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
                if (typeof setMode === 'function') setMode('es-kiche');
                document.getElementById('dynamicContentWrapper').style.opacity = '1';
                
                const ti = document.getElementById('transcriptionInput');
                const tt = document.getElementById('translationText');
                const rBox = document.getElementById('resultBox');
                
                rBox.style.display = '';
                rBox.classList.remove('hidden', 'opacity-0');
                rBox.classList.add('flex', 'opacity-100');
                ti.value = "Me duele la cabeza";
                document.getElementById('status').textContent = "¡Traducción completada!";
                
                if (typeof analyzeMedicalContext === 'function') {
                    analyzeMedicalContext("me duele la cabeza k'ax nu jolom");
                }
                
                refreshSpotlight();

                tt.textContent = "";
                let text = "K'ax nu jolom.";
                let i = 0;
                let ival = registerInterval(() => {
                    tt.textContent += text.charAt(i);
                    i++;
                    if(i >= text.length) {
                        clearInterval(ival);
                        playTTS('/tts/quc', text);
                    }
                }, 40);
            },
            onDeselected: () => { clearTourTimeouts(); audioTour.pause(); }
        },
        {
            element: '#summaryFabContainer',
            popover: {
                title: 'Botón de Resumen Clínico',
                description: 'Tras realizar consultas médicas, este botón flotante aparece para generar un registro ordenado del caso.',
                side: "left",
                align: 'start'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">shield_locked</span> Resumen Clínico';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
                
                const fab = document.getElementById('summaryFabContainer');
                fab.classList.remove('translate-y-24', 'opacity-0', 'pointer-events-none');
                fab.classList.add('translate-y-0', 'opacity-100', 'pointer-events-auto');
                
                const badge = document.getElementById('sessionCountBadge');
                badge.textContent = "2";
                badge.classList.remove('scale-0');
                badge.classList.add('scale-100');

                refreshSpotlight(); 

                registerTimeout(() => {
                    animateCursor('#summaryFabContainer button', null, 500);
                }, 600);
            },
            onDeselected: () => {
                clearTourTimeouts();
                hideCursor();
                // We leave the FAB visible and Modal closed so Step 6 can open it naturally
            }
        },
        {
            element: '#summaryModalContent',
            popover: {
                title: 'Privacidad y Portapapeles',
                description: 'Al abrir el panel, la IA redacta el cuadro clínico. Puedes copiarlo para pegarlo en tu expediente médico y toda la información se elimina del dispositivo inmediatamente después para proteger la privacidad.',
                side: "left",
                align: 'start'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">summarize</span> Privacidad y Portapapeles';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                
                // Force open modal smoothly
                const modal = document.getElementById('summaryModal');
                const content = document.getElementById('summaryModalContent');
                modal.classList.remove('opacity-0', 'pointer-events-none');
                content.classList.remove('scale-95');
                content.classList.add('scale-100');
                
                refreshSpotlight();

                registerTimeout(() => {
                    animateCursor('#btnGenerateSummary', () => {
                        const btn = document.getElementById('btnGenerateSummary');
                        const txt = document.getElementById('summaryTextarea');
                        btn.disabled = true;
                        btn.textContent = "Generando...";
                        txt.value = "Conectando con IA...";
                        
                        registerTimeout(() => {
                            btn.disabled = false;
                            btn.textContent = "Generar Ahora";
                            txt.value = "Paciente refiere dolor agudo en la zona abdominal y presenta cefalea leve. Se recomienda evaluación general.";
                            
                            registerTimeout(() => {
                                const cBtn = document.querySelector('#summaryModalContent div.flex.gap-3 button:nth-child(2)');
                                animateCursor('#summaryModalContent div.flex.gap-3 button:nth-child(2)', () => {
                                    const oldHtml = cBtn.innerHTML;
                                    cBtn.innerHTML = '<span class="material-symbols-outlined">check</span>';
                                    registerTimeout(() => cBtn.innerHTML = oldHtml, 2000);
                                }, 500);
                            }, 1000);
                        }, 1500);
                    }, 500);
                }, 800);
            },
            onDeselected: () => {
                clearTourTimeouts();
                hideCursor();
                resetTourUI();
            }
        }
    ];

    if (hasAdminControls) {
        steps.push({
            element: '.flex.gap-2',
            popover: {
                title: 'Módulos Restringidos',
                description: 'Dado que tienes permisos, aquí verás acceso al Entrenamiento de Modelos, Reglas de Escritura y Panel de Aprobaciones. \n\n¡Disfruta usando Traductor Médico K\'iche\'!',
                side: "bottom",
                align: 'end'
            },
            onPopoverRendered: (popover) => {
                popover.title.innerHTML = '<span class="material-symbols-outlined align-middle mr-2" style="font-size:1.2rem;">admin_panel_settings</span> Módulos Restringidos';
            },
            onHighlighted: () => {
                clearTourTimeouts();
                resetTourUI();
            }
        });
    }

    window.tourDriver = driver({
        showProgress: true,
        animate: true,
        allowClose: true,
        nextBtnText: 'Siguiente ➔',
        prevBtnText: '← Atrás',
        doneBtnText: '¡Entendido!',
        progressText: 'Paso {{current}} de {{total}}',
        steps: steps,
        onDestroyed: () => {
            clearTourTimeouts();
            hideCursor();
            resetTourUI();
        }
    });

    window.tourDriver.drive();
}
