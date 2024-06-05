const selectImage = document.querySelector('.select-image');
const inputFile = document.querySelector('#file');
const imgArea = document.querySelector('.img-area');
const predict = document.querySelector('.predict');

selectImage.addEventListener('click', function () {
    inputFile.click();
});

inputFile.addEventListener('change', function () {
    const image = this.files[0];
    if (image.size < 10000000) {
        const reader = new FileReader();
        reader.onload = () => {
            const allImg = imgArea.querySelectorAll('img');
            allImg.forEach(item => item.remove());
            const imgUrl = reader.result;
            const img = document.createElement('img');
            img.src = imgUrl;
            imgArea.appendChild(img);
            imgArea.classList.add('active');
            imgArea.dataset.img = image.name;
            document.getElementById("textbox").innerText = "";
        };
        reader.readAsDataURL(image);
    } else {
        alert("Image size more than 10MB");
    }
});

predict.addEventListener('click', async function () {
    const imageFile = inputFile.files[0];
    if (imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        try {
            document.getElementById("textbox").innerText = "Predicting...";
            const response = await fetch("/api/prediction", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            document.getElementById("textbox").innerText = "Result : " + result.prediction;
        } catch (error) {
            document.getElementById("textbox").innerText = "Failed to get prediction.";
        }
    } else {
        alert('Please select an image first.');
    }
});
