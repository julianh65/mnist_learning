const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

canvas.addEventListener("mousedown", () => {
  isDrawing = true;
});
canvas.addEventListener("mouseup", () => {
  isDrawing = false;
});
canvas.addEventListener("mousemove", (event) => {
  if (isDrawing) {
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(event.offsetX, event.offsetY, 10, 0, Math.PI * 2, true);
    ctx.fill();
  }
});

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
function classifyDigit() {
  const imageData = canvas.toDataURL("image/png");
  console.log(imageData);
  fetch("http://127.0.0.1:5000/classify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: imageData }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById(
        "result"
      ).textContent = `Predicted Digit: ${data.prediction}`;
    })
    .catch((error) => console.error("Error:", error));
}
