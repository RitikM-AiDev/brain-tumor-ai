import React, { useState } from "react";
import "./App.css";

function UploadImage() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        console.log("sent successful");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();

        window.URL.revokeObjectURL(url);
      } else {
        console.log("Error");
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "100px" }}>
      <h1>BRAIN-TUMOR DETECTOR</h1>
      <h2>Upload Image</h2>

      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={loading}
        />
        <br />
        <br />

        <button type="submit" disabled={loading}>
          {loading ? "Downloading..." : "Submit"}
        </button>
      </form>
    </div>
  );
}

export default UploadImage;