import React, { useState } from "react";
import YoutubeForm from "./components/YoutubeForm";
import YoutubeOutput from "./components/YoutubeOutput";
import "./App.css";

function App() {
  const [summary, setSummary] = useState(null);
  const [captions, setCaptions] = useState(null);
  const [error, setError] = useState(null);

  const handleSummary = (data) => {
    setSummary(data);
    setError(null);
  };

  const handleCaptions = (data) => {
    setCaptions(data);
    setError(null);
  };

  const handleError = (error) => {
    setSummary(null);
    setError(error);
  };

  return (
    <div className="App">
      <h1>YouTube Video Summary</h1>
      <YoutubeForm onSummary={handleSummary} onCaptions={handleCaptions} onError={handleError} />
      <div className="youtubeFormOutput">
        {summary && <YoutubeOutput summary={summary} />}
        {captions && <YoutubeOutput captions= {captions} />}
      </div>
       {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;
