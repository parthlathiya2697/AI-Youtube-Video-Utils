import React from "react";
import "./YoutubeOutput.css";

function YoutubeOutput({ summary, captions }) {
  if (!summary && !captions) return null;

  return (

    <div className="youtubeOutput">
      
      { summary && <h2>Video Summary</h2> }
      { summary && <p>{summary.data}</p> }

      { captions && <h2>Video Captions</h2> }
      { captions && <p>{captions.data}</p> }
      
    </div>
  );
}

export default YoutubeOutput;
