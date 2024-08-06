// Define the triangle vertices
const p1 = [0, 0];
const p2 = [1, 0];
const p3 = [0, 1];

// Define midpoints and barycenter
const m1 = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2];
const m2 = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2];
const m3 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
const b = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3];

// Define point classes
const T1 = [p1, p2, p3];
const T2 = [m1, m2, m3];
const T3 = [b];

function T4(a) {
    return [
        [a * p1[0] + (1 - a) * p2[0], a * p1[1] + (1 - a) * p2[1]],
        [a * p2[0] + (1 - a) * p1[0], a * p2[1] + (1 - a) * p1[1]],
        [a * p3[0] + (1 - a) * p1[0], a * p3[1] + (1 - a) * p1[1]],
        [a * p1[0] + (1 - a) * p3[0], a * p1[1] + (1 - a) * p3[1]],
        [a * p3[0] + (1 - a) * p2[0], a * p3[1] + (1 - a) * p2[1]],
        [a * p2[0] + (1 - a) * p3[0], a * p2[1] + (1 - a) * p3[1]]
    ];
}

function T5(a) {
    return [
        [a * m1[0] + (1 - a) * p1[0], a * m1[1] + (1 - a) * p1[1]],
        [a * m2[0] + (1 - a) * p2[0], a * m2[1] + (1 - a) * p2[1]],
        [a * m3[0] + (1 - a) * p3[0], a * m3[1] + (1 - a) * p3[1]]
    ];
}

function T6(a, b) {
    return [
        [b * (a * m1[0] + (1 - a) * p1[0]) + (1 - b) * (a * m2[0] + (1 - a) * p2[0]), b * (a * m1[1] + (1 - a) * p1[1]) + (1 - b) * (a * m2[1] + (1 - a) * p2[1])],
        [b * (a * m1[0] + (1 - a) * p1[0]) + (1 - b) * (a * m3[0] + (1 - a) * p3[0]), b * (a * m1[1] + (1 - a) * p1[1]) + (1 - b) * (a * m3[1] + (1 - a) * p3[1])],
        [b * (a * m3[0] + (1 - a) * p3[0]) + (1 - b) * (a * m2[0] + (1 - a) * p2[0]), b * (a * m3[1] + (1 - a) * p3[1]) + (1 - b) * (a * m2[1] + (1 - a) * p2[1])],
        [b * (a * m2[0] + (1 - a) * p2[0]) + (1 - b) * (a * m1[0] + (1 - a) * p1[0]), b * (a * m2[1] + (1 - a) * p2[1]) + (1 - b) * (a * m1[1] + (1 - a) * p1[1])],
        [b * (a * m3[0] + (1 - a) * p3[0]) + (1 - b) * (a * m1[0] + (1 - a) * p1[0]), b * (a * m3[1] + (1 - a) * p3[1]) + (1 - b) * (a * m1[1] + (1 - a) * p1[1])],
        [b * (a * m2[0] + (1 - a) * p2[0]) + (1 - b) * (a * m3[0] + (1 - a) * p3[0]), b * (a * m2[1] + (1 - a) * p2[1]) + (1 - b) * (a * m3[1] + (1 - a) * p3[1])]
    ];
}



function draw(ctx, specs, a) {
    // Clear the canvas and set background to white    
    ctx.clearRect(-10, -10, ctx.canvas.width+10, ctx.canvas.height+10);

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Set line properties for the triangle
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 0.005;  // Adjust this value as needed

    // Draw the triangle (flipped!)
    ctx.beginPath();
    ctx.moveTo(p1[0], 1-p1[1]);
    ctx.lineTo(p2[0], 1-p2[1]);
    ctx.lineTo(p3[0], 1-p3[1]);
    ctx.closePath();
    ctx.stroke();

    // Draw points based on specs
    ctx.fillStyle = 'blue';
    let aIndex = 0;
    for (let [T, paramCount] of specs) {
        let points;
        switch (T) {
            case 1: points = T1; break;
            case 2: points = T2; break;
            case 3: points = T3; break;
            case 4: points = T4(a[aIndex]); aIndex++; break;
            case 5: points = T5(a[aIndex]); aIndex++; break;
            case 6: points = T6(a[aIndex], a[aIndex + 1]); aIndex += 2; break;
        }
        for (let point of points) {
            ctx.beginPath();
            ctx.arc(point[0], 1-point[1], 0.01, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

const configurations = {
    default: {
        name: "Order 2",
        specs: [
            [1, 0], // Vertices (T1)
            [3, 0], // Trig midpoint (T1)
        ],
        a: []
    },
    config1: {
        name: "Order 4",
        specs: [
            [1, 0], // Vertices (T1)
            [2, 0], // Edge mitpoints (T4)
            [5, 1], // Interior class, type 1 (T5)
        ],
        a: [0.377160969392890078542308748058833783749855936239417087476616327085870339078554]
    },
    config2: {
        name: "Order 6",
        specs: [
            [1, 0], // Vertices (T1)
            [4, 1], // Edge class (T4)
            [5, 1], [5, 1], // Interior class, type 1 (T5)
        ],
        a: [0.3077459416259916461046162842462509600382936084322436979861164895439910651052515,
            0.8506802519794943040508623079882848864972963422938270486893584565631778072940569,
            0.2372273727931857363813267950086108322143854707884946004397634288770915154720492]
    },
    config3: {
        name: "Order 8",
        specs: [
            [1, 0], // Vertices (T1)
            [2, 0], // Edge midpoints
            [4, 1], // Edge class (T4)
            [3, 0], // Trig midpoint
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [0.800367892880542780859316780430570086983414475083430734497322780980954417519306,
            0.1609918383400748888920915101703617232494704551956004460871287522596963203890272,
            1.178990790353538136009218531055766514981000410132387662338316292442963131541253,
            0.6270816346026514964714447471534388904613573650529973305362107146733497580638089]
    },
    config4: {
        name: "Order 10",
        specs: [
            [1, 0], // Vertices (T1)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [0.1322645816327139853538882200436473589432092214523613042891785652912853424033825,
            0.3632980741536860457055063361841810532259840590132217699041213906669667508848856,
            0.5137182145239181521277816629644033613478889432665964983204021026648219094966786,
            0.115055368822820211321635035530863800881213087497156161764637485204809166321383,
            0.9156736761583222038770064348481366069448359711167989051103586524373708347553003,
            0.1563851672510340439977719569396516562552840409095293866190088949247856013241157,
            0.8134250274303948218471218265542466604673601990058271109937427943875009474310373]
    },
    config5: {
        name: "Order 12",
        specs: [
            [1, 0], // Vertices (T1)
            [2, 0], // Edge midpoints
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [0.08241681507823011945294286373531904405845952423134596282614025628517770576804981,
            0.7336586356282496528158148279730068963729017833255884881520737441032971857121487,
            0.3968992432090585282173714149068856270044609413319084636886713959115893061823328,
            0.08483206729954031152189230763403608792291905858396488199050836173441803150625181,
            0.7897555797440240196725215804242898435608124756653441874097675727603200540085861,
            0.1324011112438714654919235821499105342443585799348871082110686313515405366807619,
            0.3618924416944691121874987235030598557249083247797107107788101611925815667687246,
            0.1147700956805489009249444304666597162954323750142438600326002054576767855213305,
            0.127964059325616321114149468299952478585934750904417616264837824112179310167174]
    },
    config6: {
        name: "Order 14",
        specs: [
            [1, 0], // Vertices (T1)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [ 0.6210448422613127,
            0.07972789185817206,
            0.7986370960715439,
            0.5611177925929208,
            0.9599692164209439,
            0.0663324701435207,
            0.33433070672479104,
            0.09983680739923065,
            0.9032546144314979,
            0.09867362791330596,
            0.7281109491586926,
            1.0329436596868007,
            0.31597553195402994]
    }
};

function calculatePoints(specs, a) {
    let points = [];
    let aIndex = 0;
    
    for (let [T, paramCount] of specs) {
        switch (T) {
            case 1:
                points = points.concat(T1);
                break;
            case 2:
                points = points.concat(T2);
                break;
            case 3:
                points = points.concat(T3);
                break;
            case 4:
                points = points.concat(T4(a[aIndex]));
                aIndex++;
                break;
            case 5:
                points = points.concat(T5(a[aIndex]));
                aIndex++;
                break;
            case 6:
                points = points.concat(T6(a[aIndex], a[aIndex + 1]));
                aIndex += 2;
                break;
        }
    }
    
    return points;
}

function populateDropdown() {
    const select = document.getElementById('specSelect');
    select.innerHTML = ''; // Clear existing options
    
    for (const [key, config] of Object.entries(configurations)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = config.name;
        select.appendChild(option);
    }
}

function initializeWithMathJax() {
    MathJax.startup.promise.then(() => {
        populateDropdown();
        drawSelected(ctx);
    });
}

document.addEventListener('DOMContentLoaded', initializeWithMathJax);


function updateInfo(config) {
    const specInfo = document.getElementById('specInfo');
    const vectorInfo = document.getElementById('vectorInfo');
    const pointsInfo = document.getElementById('pointsInfo');
    
    // Format specs as a compact 2xN matrix
    let formattedSpecs = config.specs.reduce((acc, spec) => {
        acc[0].push(spec[0]);
        acc[1].push(spec[1]);
        return acc;
    }, [[], []]);

    let specString = `\\begin{bmatrix}
    ${formattedSpecs[0].join(' & ')} \\\\
    ${formattedSpecs[1].join(' & ')}
    \\end{bmatrix}`;
    
    // Format vector a
    let vectorString = `\\begin{matrix} ${config.a.join(' \\\\ ')} \\end{matrix}`;
    
    // Calculate and format points
    let points = calculatePoints(config.specs, config.a);
    let pointsString = points.map((p, i) => `P_{${i+1}} = (${p[0].toFixed(8)}, ${p[1].toFixed(8)})`).join(' \\\\ ');
    
    specInfo.innerHTML = `<h3>Specs for ${config.name}:</h3>$$${specString}$$`;
    vectorInfo.innerHTML = `<h3>Vector a for ${config.name}:</h3>$$${vectorString}$$`;
    pointsInfo.innerHTML = `<h3>Points for ${config.name}:</h3>$$\\begin{align*}${pointsString}\\end{align*}$$`;

    // Trigger MathJax to typeset the new content
    MathJax.typesetPromise([specInfo, vectorInfo, pointsInfo]).catch((err) => console.log('MathJax typesetting failed: ' + err.message));
}

function drawSelected(ctx) {
    const select = document.getElementById('specSelect');
    const selectedConfig = configurations[select.value];
    
    updateInfo(selectedConfig);
    draw(ctx, selectedConfig.specs, selectedConfig.a);
}

// Set up the canvas
const canvas = document.getElementById('triangleCanvas');
const ctx = canvas.getContext('2d');

// Set canvas size
canvas.width = 500;
canvas.height = 500;

// Scale and translate the context to fit the triangle
ctx.scale(400, 400);
ctx.translate(0.1, 0.1);

// Set up the dropdown event listener
const select = document.getElementById('specSelect');
select.addEventListener('change', drawSelected);



// Make sure the canvas is available before trying to get its context
document.addEventListener('DOMContentLoaded', (event) => {
    const canvas = document.getElementById('triangleCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = 500;
        canvas.height = 500;

        // Scale and translate the context to fit the triangle
        ctx.scale(400, 400);
        ctx.translate(0.1, 0.1);

        // Set up the dropdown event listener
        const select = document.getElementById('specSelect');
        if (select) {
            select.addEventListener('change', () => drawSelected(ctx));

            // Initial draw
            drawSelected(ctx);
        } else {
            console.error('Dropdown menu not found');
        }
    } else {
        console.error('Canvas element not found');
    }
});




// Initial draw
drawSelected();
