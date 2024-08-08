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
        a: [ 0.6210448422613125942170433019897433243027482598742658653862481265180586650123877,
            0.07972789185817234961642169407997058268526583049564610269494367078048400909698998,
            0.7986370960715435347144142839995913887306097083906321354983657654808684181778722,
            0.5611177925929207588292702740864206566459848291743871765163728736140644922129322,
            0.9599692164209438828074976110052227808439508002411262475960122440486067801491928,
            0.06633247014352065906413598030085613222155185561658038629671329130986642119500806,
            0.3343307067247910280055434458787071333143759525959957706576779214238862316037228,
            0.09983680739923061856787629327107723847724931922625328720775778560503922508485003,
            0.9032546144314978309614034134534899050928336666989309262000553050619463590537733,
            0.098673627913305978592434634115609331774468203500912912143787518879697480489469,
            0.728110949158692572885074217335401004553598921157664562876023590982614915202393,
            1.03294365968680069346728076206226775218341942721867165675984382843548203723304,
            0.3159755319540299543233680103499439922279952963609570701172332560340826802494863]
    },
    config7: {
        name: "Order 16",
        specs: [
            [1, 0], // Vertices (T1)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [2, 0], // Edge class (T4)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [0.8425091314698993486123905153612640588891622173891286020893362416631174583389535,
            0.6949176801528631593381856056756541532430840523288657160356682879787925605309801,
            0.06014551285115061380288544774618444272822243048515844667836791401623852770512386,
            0.7608692813931116227949760108583722442500529385183998025031203662271292345718708,
            0.480479350537909419623840400827624059083689779893326524916270977710506338532866,
            0.05128162047782862435145596609953120449919709471708357636778200159260087329024858,
            0.2692849896531915578388146147043470737778093743725325411068648114887133550160312,
            0.8828510628585303471282702155941680254106923052724471233053260855029329076540273,
            0.08139286964794953229877211260657370057118148115665692848987117100775164171044678,
            0.2112496012057297777439712145930152306983859473863795804586828500506025194816036,
            0.2466319822858454289595243165534586427129249050394245126268501700024275936017323,
            0.7541945351195432442752932352484073150754819689370655204355415410223830917363402,
            0.07262588913486557586010419329245855119915489406378663205313620756250973030285028,
            0.6020754859768345654231316390491057893318689094370976301460049389989796237238874,
            0.207983451806118440290243607902854243181583723716587954712549288157046335026004,
           -0.09392004467636680538041810168700589141495951826060772461489151376593522362733116],
        w: [7.231791524510943e-5,
            0.001811313401313938,
            0.001590576198932087,
            0.0027579504603657606,
            0.001490958718700355,
            0.006892755307739218,
            0.019702161005143666,
            0.016437507138291334,
            0.024332575446248565,
            0.018654394665537078,
            0.01960711458105574,
            0.0026554694866164665,
            0.021586516994465656,
            0.012619175246335032,
            0.02830036920367197,
            0.013350146557782416,
            0.021300224661661416,
            0.0076822874504994376]
    },
    config8: {
        name: "Order 18",
        specs: [
            [1, 0], // Vertices (T1)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [4, 1], // Edge class (T4)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [5, 1], // Interior class, type 1 (T5)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
            [6, 2], // Interior class, type 2 (T6)
        ],
        a: [0.22602274479966727,
            0.05554275033556339,
            0.8809651342369424,
            0.38610851964110793,
            0.8215824924982053,
            0.40470602317511895,
            0.5778432879613025,
            0.2306725741198919,
            0.9808841598152929,
            0.0430455226280874,
            0.32783005074502947,
            0.32610232636680003,
            0.1539705448108261,
            0.5948746214874675,
            0.19304049368084872,
            0.8026525798695466,
            0.052427440911469204,
            0.31819283585385766,
            0.06892420195266408,
            0.05979099891334915,
            0.06595443915802122,
            0.8314386543713359]
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
