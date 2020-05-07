let inputs = [];


function setup() {
  createCanvas(280, 280);
  background(255);

  let clearButton = select('#clear');
  clearButton.mousePressed(function(){
    background(255);
});

  let guessButton = select('#guess');
  guessButton.mousePressed(function() {
    let img = get(); /* get all pixels from sketch */
    img.resize(28, 28);
    img.loadPixels();
    for (let i = 0; i < 784; i++) {
      let bright = img.pixels[i*4];
      inputs[i] = (255 - bright) / 255.0;     /* inputs = array of pixel density white-black btwn 0-1, size= 784 */
    }
  });
}


function getVals(){
   return inputs;                       
}
$(document).ready(function () {
$("#guess").on("click", function() {
var js_data = JSON.stringify(getVals());
$.ajax({
url: '/',
type : 'post',
contentType: 'application/json',
dataType : 'json',
data : js_data
}).done(function(result) {
         console.log(result);
         let ans = result;
         alert(ans['answer']);
        $("#data").html(result);
        }).fail(function(jqXHR, textStatus, errorThrown) {
            console.log("fail: ",textStatus, errorThrown);
       });
   });
});


function draw() {
  strokeWeight(3);
  stroke(0);
  if (mouseIsPressed) {
     line(pmouseX, pmouseY, mouseX, mouseY);
  }
}

