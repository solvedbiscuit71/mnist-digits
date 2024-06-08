import React, { useRef, useState, useEffect } from 'react';

const Canvas = ({ onPredict }) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.lineWidth = 10;
        context.lineCap = 'round';
        context.strokeStyle = 'black';
    }, []);

    const startDrawing = (event) => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.beginPath();
        context.moveTo(
            event.nativeEvent.offsetX,
            event.nativeEvent.offsetY
        );
        setIsDrawing(true);
    };

    const finishDrawing = () => {
        setIsDrawing(false);
    };

    const draw = (event) => {
        if (!isDrawing) return;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.lineTo(
            event.nativeEvent.offsetX,
            event.nativeEvent.offsetY
        );
        context.stroke();
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
    };

    const predictDigit = () => {
        const canvas = canvasRef.current;
        const dataURL = canvas.toDataURL('image/png');
        onPredict(dataURL)
    };

    return (
        <div>
            <canvas
                ref={canvasRef}
                width="140"
                height="140"
                style={{ border: '1px solid black' }}
                onMouseDown={startDrawing}
                onMouseUp={finishDrawing}
                onMouseMove={draw}
            />
            <br />
            <button onClick={clearCanvas}>Clear</button>
            <button onClick={predictDigit}>Predict</button>
        </div>
    );
};

export default Canvas;