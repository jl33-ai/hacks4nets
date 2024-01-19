document.addEventListener('DOMContentLoaded', function() {
    const vnn_frames = [
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒▒░░░█
█░░▒▒▓▒░░░░░░░░░░░░░░░░▓▒▓░░░█
█░░░░░▒░░░▓▒▓▒▒▒▓▒▓░░░▒▒░░░░░█
█░░░▒░░▒▒░▒▒▒░░▒▒▒▒▓▒▒░░▒░░░░█
█░░▒▒▓▒░▒▒░░░░▒░░░░▒▓▒▒▓▒▓░░░█
█░░░▒▒░░░░▓▒▓▒░░▒▒▓▒░▒░░▒░░░░█
█░░░░░░░░░▒▒▒░░░▒▒▒▒▒░▒░░░░░░█
█░░▒▒▓▒░░░░░░░░░░░░░░▒▒▓▒▓░░░█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒░░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒▒░░░█
█░░▒▒▓▒░░░░░░░░░░░░░░░░▓▒▓░░░█
█░░░░░░░▒▒▓▒▓▒▒▒▓▒▓░░░▒▒░░░░░█
█░░░▒░░░▒▒▒▒▒░░░▒▒▒▓▒▒░░▒░░░░█
█░░▒▒▓▒▒░░░░░░░░░░░▒▓▒▒▓▒▓░░░█
█░░░▒▒░▒▒▒▓▒▒░░░▒▒▓▒░▒░░▒░░░░█
█░░░░░░░▒▒▒▒▒░░░▒▒▒▒▒░▒░░░░░░█
█░░▒▒▓▒▒░░░░░░░░░░░░░▒▒▓▒▓░░░█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒░░░░█`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒▒░░░█
█░░▒▒▓▒░░░░░░░░░░░░░░░░▓▒▓░░░█
█░░░░░▒▒▒▒▓▒▓░░░▒▒▓░░░▒▒░░░░░█
█░░░▒░░▒▒▓▒▒▒▒░▒▒▒▒░░▒░░▒░░░░█
█░░▒▒▓▒▒▒▒░░░░▓▒░░░▒▒░░▓▒▓░░░█
█░░░▒▒░▒▒▒▓▒▓▒░▒▓▒▓▒░░░░▒░░░░█
█░░░░░░▒▒▒▒▒▒░░░▒▒▒░░░░░░░░░░█
█░░▒▒▓▓▒░░░░░░░░░░░░░░░▓▒▓░░░█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒░░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒▒░░░█
█░░▒▒▓▒░░░░░░░░░░░░░░▒▒▓▒▓░░░█
█░░░░░▒▒▒▒▓▒▓▒▒▒▓▒▓▒▒░▒▒░░░░░█
█░░░▒░░▒▒▒▒▒▒▒░▒▒▒▒▓▒▒░░▒░░░░█
█░░▒▒▓▒▒▒▒░░░░▓▒░░░▒▓▒▒▓▒▓░░░█
█░░░▒▒░▒▒▒▓▒▓▒░▒▓▒▓▒░▒░░▒░░░░█
█░░░░░░░▒▒▒▒▒░░░▒▒▒▒▒░▒░░░░░░█
█░░▒▒▓▒▒░░░░░░░░░░░░░▒▒▓▒▓░░░█
█░░░▒▒░░░░░░░░░░░░░░░░░▒▒░░░░█
`

    ];

    const image_frames = [
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░▒░░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒░░░░░░░▓▓▓▓▓▓██▒░▒▓▓▒░█
█░░░░░░▒▒▒▒░░░███▓▓▓▓▓░░▒▓▓▒░█
█░░▒▒▒▒▒▒▒▒░░░█████▒░░░░░▒▒░░█
█░▒▒░░░░░░▒▒░░███▒▒▓███████▒░█
█░░▒▒▒▒▒▒▒▒▒░░▒▒▒██▓▒▒███▓▓░░█
█░░░▒▒▒░░░░▒░░▒▒▓██▓░░▓▓▓▓▓░░█
█░░▒░░░▒▒▒░░░░▓▓▓██▒░░░░▒██▒░█
█░░▒▒▒░░░░░░░░░░░░░░░░░░░░░░░█
`, 
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░█
█░░░░▒▒▒▒▒░░░░░░▓██▒░▒█████░░█
█░░░▒▒▒▒░░▒▒░░░░▓██▓▓▓█████░░█
█░░░░▒▒▒▒▒▒▒▒░▒▒▓█████▓▓▓▓▓░░█
█░░▒░░░░░░░░░░█████▓▓▒░░░░░░░█
█░░░▒▒▒░░░░░░░███▓▓████████░░█
█░░░▒▒▒▒▒▒▒▒░░█████████████░░█
█░░░░░░▒▒▒▒▒░░█████▓▓▓█████░░█
█░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░█
`, 
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░▒░░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒▒░░░░░░░░▓▓▓▒░▒██▓▒░░░█
█░░░░░░▒▒▒▒▒░░░▒███▒▒▒██▓▒▒░░█
█░░▒▒▒▒▒▒▒▒░░░▓▓███▒▓▓██▓▓▓░░█
█░░░░░░░░░░░░░▒░░░░░▒▒░░░░░░░█
█░░░░░░░░░░▒░▒██▓░░░▒▒░░▓██░░█
█░░▒▒▒▒▒▒▒▒░░▒███▓▓▓▓▓▓▓███░░█
█░░▒▒░░░░░░░░▒█████████████░░█
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░█
█░░░░▒▒▒▒▒░░░░░░▓▓▓░░▒██▒░░░░█
█░░░░░░░░░▒▒░░░░███░░▒██▓░░░░█
█░░▒▒▒▒▒▒▒▒░░░░░███░░▒██▒░░░░█
█░░▒▒▒░░░░░▒░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒░▒▒▒▒░▒██▒░░░░░░░▓██░░█
█░░▒▒▒▒▒▒▒▒▒▒▒███▓▓▓▓▓▓▓███░░█
█░░░░░░▒▒▒▒░░▒█████████████░░█
█░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒░░░░░░░░░▓▓▓░░▒██▒░░░░█
█░░░░░░░▒▒▒▒░░░░███░░▒██▓░░░░█
█░░░░░░░░░░░░░░░███░░▒██▒░░░░█
█░░▒▒▒░░░░░░░░░░░░░░░░░░░░░░░█
█░░░░░▒▒▒▒▒▒░▒██▒░░░░░░░▓██░░█
█░░▒▒▒▒▒░░▒▒░▒███▓▓▓▓▓▓▓███░░█
█░░░░░░░░░░░░▒█████████████░░█
█░░▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░█
`,
    ]

    const llm_frames = [
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▓████████▒░░░░░░▓██████▓░░█
█░▒██▓██████▓░░░░░▒█▓▓▓▓▓▓█▒░█
█░▒██▓▓█████▓░░░░░▒█▓▓▓▓▓▓█▒░█
█░░▓████████▒░░░░░▒█▒░░░░▒█▒░█
█░░░░░░░░░░░░░░░░░▒████████▒░█
█░░█████████▒░░░░░▒████████▒░█
█░▒██▒███████▒▒▒▓▒▒█▓▒▒▒▒▒█▒░█
█░▒██▓██████▓░░░▒░▒████████▒░█
█░░▓███████▓░░░░░░░▒▓████▓▒░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▓████████▒░░░░░░▓██████▓░░█
█░▒██▓█▓▓███▓░░░▒░▒█▓▓▓▓▓▓█▒░█
█░▒██▓▓▓▓████▒▒▒▒░▒█▓▓▓▓▓▓█▒░█
█░░▓████████▒░░░░░▒█▒░░░░▒█▒░█
█░░░░░░░░░░░░░░░░░▒████████▒░█
█░░█████████▒░░░░░▒█▒▒▒▒▒▒█▒░█
█░▒██▒█▓▓███▓░░░░░▒████████▒░█
█░▒██▓█▓████▓░░░░░▒████████▒░█
█░░▓███████▓░░░░░░░▒▓████▓▒░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▓████████▒░░░░░░▓██████▓░░█
█░▒██▓█▓▓███▓░░░▒░▒████████▒░█
█░▒██▓▓▓▓████▒▒▒▒░▒████████▒░█
█░░▓████████▒░░░░░▒█▒░░░░▒█▒░█
█░░░░░░░░░░░░░░░░░▒████████▒░█
█░░█████████▒░░░░░▒█▒▒▒▒▒▒█▒░█
█░▒██▒█▓▓█▒██▒▒▒▓▒▒█▓▒▒▒▒▒█▒░█
█░▒██▓█▓██▓█▓░░░▒░▒████████▒░█
█░░▓███████▓░░░░░░░▒▓████▓▒░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▓████████▒░░░░░░▓██████▓░░█
█░▒█████████▓░░░░░▒█▓▓▓▓▓▓█▒░█
█░▒█████████▓░░░░░▒█▓▓▓▓▓▓█▒░█
█░░▓████████▒░░░░░▒████████▒░█
█░░░░░░░░░░░░░░░░░▒████████▒░█
█░░█████████▒░░░░░▒█▒▒▒▒▒▒█▒░█
█░▒█████████▓░░░░░▒█▓▒▒▒▒▒█▒░█
█░▒█████████▓░░░░░▒████████▒░█
█░░▓███████▓░░░░░░░▒▓████▓▒░░█
`,
    ]

    const ml_frames = [
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░▒▓▓▒░░░░░░░░░░░░░░░░░░░░█
█░░█░░▒▒░░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░▓▒▓░░░░░░░░░░░░░░░░█
█░░█░░░░░░▒▒░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░▓▒▓░░░░░░░░░░░░░░░░░░░█
█░░█░░░▒▒▒░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░█░░░░░░▒▒▓▒░░░░░▒░░░░░░░░░█
█░░█░░░░░░░▒▒░░░░░▓▒▓░░░░░░░░█
█░░█░░░▒░░░░░░░░░░░▒░░░░░░░░░█
█░░█░░▓▒▓░░░░░░░░░░░░░░░░░░░░█
█░░█░░▒▒░░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒░░░░░░░░░░░░░░░░░▒▒▒░░░░░█
█░░█░░░░░░░░░░░░░░░░░▒▒▓░░░░░█
█░░█░░░░░░░▒░░░░░░░░░░░░░░░░░█
█░░█░░░░░░▓▒▓░░░░░░░░░░░░░░░░█
█░░█░░░░░░░▒░░░░░░▒▒▓░░░░░░░░█
█░░█░░░▒░░░░░░░░░░▒▒▒░░░░░░░░█
█░░█░░▒▒▓▒░░░░░░░░░░░░░░░░░░░█
█░░█░░░▒▒░░░░░░░░░░░░░░░░░░░░█
█░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░█
`,
`█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
█░░▒░░░░░░░░░░░░░░░░░▒▒▒▒▓█▓▒█
█░░█░░░░░░░░░░░░░░░░░▓▓█▓▒░░░█
█░░█░░░░░▒▒▒░░░░░░▒▓█▓▒░░░░░░█
█░░█░░░░░▒▒▒░░░▒▓█▓▒░░░░░░░░░█
█░░█░░░░░░░░▒▓█▓▒░░▒▒░░░░░░░░█
█░░█░▒▒░░▒▓█▓▒░░░░▒▓▒▒░░░░░░░█
█░░█░▓▒██▓▒░░░░░░░░░░░░░░░░░░█
█░░█▓█▓▒░░░░░░░░░░░░░░░░░░░░░█
█▒█▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░█
`
    ]


    let cf0 = 0;
    let cf1 = 0;
    let cf2 = 0;
    let cf3 = 0;

    function updateAscii() {
        const asciiElement = document.getElementById('ascii-image');
        asciiElement.textContent = image_frames[cf0];
        cf0 = (cf0 + 1) % image_frames.length;
        
        const asciiElement1 = document.getElementById('ascii-vnn');
        asciiElement1.textContent = vnn_frames[cf1];
        cf1 = (cf1 + 1) % vnn_frames.length;

        const asciiElement2 = document.getElementById('ascii-llm');
        asciiElement2.textContent = llm_frames[cf2];
        cf2 = (cf2 + 1) % llm_frames.length;

        const asciiElement3 = document.getElementById('ascii-ml');
        asciiElement3.textContent = ml_frames[cf3];
        cf3 = (cf3 + 1) % ml_frames.length;
    }

    setInterval(updateAscii, 250); // Adjust the interval to control animation speed
});
