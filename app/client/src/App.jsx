import { useContext, useState } from 'react';
import { APIContext } from './Context';
import Canvas from './component/Canvas'

const dataURLtoBlob = (dataurl) => {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
};

function App() {
  const apiURL = useContext(APIContext);
  const [prediction, setPrediction] = useState('');

  async function handlePredict(dataURL) {
    try {
      const formData = new FormData();
      formData.append('file', dataURLtoBlob(dataURL));

      const response = await fetch(`${apiURL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok)
        throw new Error('Network responsed with status != 200');
      
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error('Unexpected Error during prediction', err);
    }
  }

  return (
    <>
      <Canvas onPredict={handlePredict} />
      <p>
        Prediction: {prediction}
      </p>
    </>
  )
}

export default App
