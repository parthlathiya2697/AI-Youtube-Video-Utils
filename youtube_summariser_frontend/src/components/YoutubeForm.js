import React, { useState } from "react";
import axios from "axios";
import "./YoutubeForm.css";

function YoutubeForm({ onSummary, onCaptions, onError }) {
  const [url, setUrl] = useState("");

  const get_video_captions = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://0.0.0.0:80/fetch_video_captions", { url });
      let response_data = response.data;
      console.log(response_data)
      onCaptions(response_data);
    } catch (error) {
      onError(error.message);
    }
  };
  
  const get_video_summary = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://0.0.0.0:80/fetch_video_summary", 
        { url: url, summarisation_technique: 'summarise_text_openai' }
      );
      let response_data = response.data;
      console.log(response_data)
      onSummary(response_data);
    } catch (error) {
      onError(error.message);
    }
  };

  return (
    <form onSubmit={get_video_summary}>
      <input type="text" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Enter YouTube Video URL" />
      <input type="submit" value="Get Summary" />
      <button onClick={get_video_captions}> Get Captions </button>
      
    </form>

  );
}

export default YoutubeForm;
