/* ----- NAVIGATION BAR FUNCTION ----- */
function myMenuFunction() {
  var menuBtn = document.getElementById("myNavMenu");

  if (menuBtn.className === "nav-menu") {
    menuBtn.className += " responsive";
  } else {
    menuBtn.className = "nav-menu";
  }
}

/* ----- ADD SHADOW ON NAVIGATION BAR WHILE SCROLLING ----- */
window.onscroll = function () {
  headerShadow();
};

function headerShadow() {
  const navHeader = document.getElementById("header");

  if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50) {
    navHeader.style.boxShadow = "0 1px 6px rgba(0, 0, 0, 0.1)";
    navHeader.style.height = "70px";
    navHeader.style.lineHeight = "70px";
  } else {
    navHeader.style.boxShadow = "none";
    navHeader.style.height = "90px";
    navHeader.style.lineHeight = "90px";
  }
}

/* ----- TYPING EFFECT ----- */
var typingEffect = new Typed(".typedText", {
  strings: [
    "AI / ML Enthusiast",
    "Software Engineer",
    "Data Scientist",
    "Full-Stack Developer",
    "Graduate Student",
  ],
  loop: true,
  typeSpeed: 100,
  backSpeed: 80,
  backDelay: 2000,
});

/* ----- ## -- SCROLL REVEAL ANIMATION -- ## ----- */
const sr = ScrollReveal({
  origin: "top",
  distance: "80px",
  duration: 2000,
  reset: true,
});

/* -- HOME -- */
sr.reveal(".featured-text-card", {});
sr.reveal(".featured-name", { delay: 100 });
sr.reveal(".featured-text-info", { delay: 200 });
sr.reveal(".featured-text-btn", { delay: 200 });
sr.reveal(".social_icons", { delay: 200 });
sr.reveal(".featured-image", { delay: 300 });

sr.reveal(".timeline", { interval: 200 });
/* -- PROJECT BOX -- */
sr.reveal(".project-box", { interval: 200 });

/* -- Certification BOX -- */
sr.reveal(".certificate-box", { interval: 200 });

/* -- HEADINGS -- */
sr.reveal(".top-header", {});

/* ----- ## -- SCROLL REVEAL LEFT_RIGHT ANIMATION -- ## ----- */

/* -- ABOUT INFO & CONTACT INFO -- */
const srLeft = ScrollReveal({
  origin: "left",
  distance: "80px",
  duration: 2000,
  reset: true,
});

srLeft.reveal(".about-info", { delay: 100 });
srLeft.reveal(".contact-info", { delay: 100 });

/* -- ABOUT SKILLS & FORM BOX -- */
const srRight = ScrollReveal({
  origin: "right",
  distance: "80px",
  duration: 2000,
  reset: true,
});

srRight.reveal(".skills-box", { delay: 100 });
srRight.reveal(".form-control", { delay: 100 });

/* ----- CHANGE ACTIVE LINK ----- */

const sections = document.querySelectorAll("section[id]");

function scrollActive() {
  const scrollY = window.scrollY;

  sections.forEach((current) => {
    const sectionHeight = current.offsetHeight,
      sectionTop = current.offsetTop - 50,
      sectionId = current.getAttribute("id");

    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      document
        .querySelector(".nav-menu a[href*=" + sectionId + "]")
        .classList.add("active-link");
    } else {
      document
        .querySelector(".nav-menu a[href*=" + sectionId + "]")
        .classList.remove("active-link");
    }
  });
}

window.addEventListener("scroll", scrollActive);

// JavaScript to handle the modal
document.addEventListener("DOMContentLoaded", function () {
  const modal = document.getElementById("project-modal");
  const closeBtn = document.querySelector(".close-btn");
  const detailsBtns = document.querySelectorAll(".details-btn");

  // Project data (can be fetched from an API or hardcoded)
  const projects = {
    1: {
      name: `<h5> SmartPDF Chat: AI-Powered Document Analysis </h5>
        <br>`,
      images: ["assets/images/SmartPDF-project/6.png"],
      description: `
            <h3>Project Overview</h3>
            <p>I developed <strong>SmartPDF Chat</strong>, an advanced Python-based web application that enables users to interactively query multiple PDF documents using natural language processing (NLP). This tool allows users to upload PDF files, ask questions in plain English, and receive precise, contextually relevant answers derived exclusively from the content of the uploaded documents. By leveraging cutting-edge AI technologies, SmartPDF Chat ensures accurate and document-specific responses, making it an invaluable tool for research, business, legal, and educational applications.</p>
            <br>
            <h3>Key Features</h3>
            <ul>
                <li><strong>Natural Language Querying</strong>: Users can ask questions in natural language and receive answers directly from the content of uploaded PDFs.</li>
                <li><strong>Multi-PDF Support</strong>: The application supports querying across multiple PDF documents simultaneously.</li>
                <li><strong>Context-Aware Responses</strong>: Ensures responses are generated exclusively from the uploaded documents, guaranteeing accuracy and relevance.</li>
                <li><strong>User-Friendly Interface</strong>: Built with Streamlit, the application provides an intuitive and seamless user experience.</li>
            </ul>
            <br>
            <h3>How It Works</h3>
            <ol>
                <li><strong>Text Extraction</strong>: The application uses <strong>PyPDF2</strong> to extract text from uploaded PDF documents.</li>
                <li><strong>Text Chunking</strong>: Extracted text is divided into smaller, manageable chunks for efficient processing.</li>
                <li><strong>Embedding Generation</strong>: <strong>Sentence-transformers</strong> and <strong>InstructorEmbedding</strong> are used to create vector representations (embeddings) of the text chunks.</li>
                <li><strong>Similarity Matching</strong>: <strong>Faiss-cpu</strong> (Facebook AI Similarity Search) compares user queries with the embeddings to identify the most semantically relevant text chunks.</li>
                <li><strong>Response Generation</strong>: The <strong>google/flan-t5-xxl</strong> language model processes the selected text chunks to generate coherent and contextually accurate responses.</li>
                <li><strong>User Interaction</strong>: The <strong>Streamlit</strong> framework provides a clean and interactive interface for users to upload documents, ask questions, and view results.</li>
            </ol>

            <h3>Technologies Used</h3>
            <ul>
                <li><strong>Programming Language</strong>: Python</li>
                <li><strong>Libraries & Frameworks</strong>: LangChain, PyPDF2, Streamlit, Sentence-transformers, InstructorEmbedding, Faiss-cpu</li>
                <li><strong>Language Model</strong>: google/flan-t5-xxl</li>
                <li><strong>Deployment</strong>: Streamlit (for web-based interaction)</li>
            </ul>

            <h3>Applications & Benefits</h3>
            <ul>
                <li><strong>Research & Academia</strong>: Quickly extract key information from academic papers, theses, and research documents.</li>
                <li><strong>Business</strong>: Efficiently navigate through contracts, reports, and manuals to find specific details.</li>
                <li><strong>Legal</strong>: Instantly access pertinent information from legal documents and case files.</li>
                <li><strong>Education</strong>: Facilitate learning by interacting with study materials and textbooks.</li>
            </ul>
            <br>
            <h3>Key Achievements</h3>
            <ul>
                <li>Designed and implemented an end-to-end pipeline for text extraction, chunking, and vector representation.</li>
                <li>Integrated advanced NLP models (flan-t5-xxl) and similarity search algorithms (Faiss-cpu) to achieve precise and context-aware response generation.</li>
                <li>Delivered a user-friendly web application using Streamlit, enabling seamless interaction with the tool.</li>
            </ul>
            
            
        `,
    },

    2: {
      name: `<h5> Stock Analytics Web Application </h5>
        <br>`,
      images: ["assets/images/stockweb.png"],
      description: `
        <h3>Project Overview</h3>
        <p>Developed a full-stack web application for <strong>real-time stock analysis</strong>, enabling users to search, track, and trade stocks using <strong>Finnhub</strong> and <strong>Polygon.io</strong> APIs. The platform features <strong>dynamic data visualization</strong>, <strong>responsive design</strong>, and secure <strong>portfolio management</strong>. Built with <strong>React, Node.js, and MongoDB</strong>, and deployed on <strong>AWS EC2</strong> for scalable cloud performance.</p>

        <h3>Key Features</h3>
        <ul>
        <li><strong>Real-Time Stock Search:</strong> Integrated <strong>Finnhub API</strong> for autocomplete suggestions, company profiles, and live stock quotes.</li>
        <li>Responsive UI with <strong>React and Bootstrap</strong> for seamless search and filtering.</li>
        <li>Implemented <strong>debouncing</strong> for API calls to optimize performance during user input.</li>
        <li><strong>Interactive Charts:</strong> Visualized historical stock trends using <strong>Highcharts</strong>, displaying <strong>SMA (Simple Moving Average)</strong> and volume-price data over customizable timeframes (6M, 3M, 1M).</li>
        <li>Synced <strong>real-time updates</strong> using WebSocket-like polling for market-open scenarios.</li>
        <li><strong>Portfolio & Watchlist Management:</strong> Tracked stocks with buy/sell functionality, dynamic profit/loss calculations, and a virtual wallet (<strong>$25k initial balance</strong>).</li>
        <li>Stored user data in <strong>MongoDB Atlas</strong> for persistent watchlists and transaction history.</li>
        <li>Implemented modals for <strong>transaction validation</strong> with error handling (e.g., insufficient funds).</li>
        <li><strong>News Integration & Social Sharing:</strong> Aggregated company-specific news with modal popups for detailed articles.</li>
        <li>Enabled sharing on <strong>Twitter and Facebook</strong> using SDKs and dynamic URL parameters.</li>
        <li><strong>Responsive Design:</strong> Optimized for mobile/desktop using <strong>Bootstrap grids</strong> and React’s responsive components.</li>
        <li>Tested cross-device compatibility using <strong>Chrome’s Responsive Design Mode</strong>.</li>
        </ul>

        <h3>Technologies Used</h3>
        <ul>
        <li><strong>Frontend:</strong> React, React Router, Axios, Bootstrap 5, Highcharts, Material-UI.</li>
        <li><strong>Backend:</strong> Node.js, Express.js, RESTful APIs.</li>
        <li><strong>Database:</strong> MongoDB Atlas (NoSQL).</li>
        <li><strong>APIs:</strong> Finnhub (stock data), Polygon.io (historical prices), Facebook/Twitter SDKs.</li>
        <li><strong>Deployment:</strong> AWS EC2</li>
        </ul>

        <h3>Key Achievements</h3>
        <ul>        
        <li>Enhanced user engagement with <strong>live charts</strong> and auto-updating portfolio metrics using React state management.</li>
        <li>Achieved <strong>100% mobile responsiveness</strong> using Bootstrap’s grid system and React conditional rendering.</li>
        <li>Secured <strong>MongoDB Atlas</strong> integration for real-time data synchronization.</li>
        </ul>      

        <p><strong>Note:</strong> The source code for this project is in a <strong>private GitHub repository</strong>. If you would like to <strong>view</strong> the code, please contact me via <a href="mailto:parthdpraja@gmail.com">email</a> or LinkedIn.</p>

        `,
    },

    3: {
      name: `<h5> Duo-Othello AI Agent: Intelligent Game-Playing Agent for Competitive Board Strategy </h5>
          <br>`,
      images: ["assets/images/othello.png"],
      description: `
         <h3>Project Overview</h3>
    <p>Developed a Python-based <strong>AI agent</strong> for <strong>Duo-Othello</strong>, a 12x12 variant of Reversi/Othello, designed to compete against reference agents in a <strong>time-constrained</strong> environment. The agent leverages <strong>alpha-beta pruning</strong> and <strong>adaptive depth selection</strong> to optimize move decisions under strict time limits (e.g., 300 seconds total). Achieved a <strong>90% win rate</strong> against a minimax reference agent.</p>
<br>
    <h3>Key Features</h3>
    <ul>
        <li><strong>Alpha-Beta Pruning:</strong> Optimized minimax algorithm with alpha-beta pruning to reduce search space by 40%, enabling deeper exploration of game trees.</li>
        <li><strong>Dynamic Time Management:</strong> Adjusted search depth (1–4 levels) based on remaining playtime, prioritizing speed in low-time scenarios.</li>
        <li><strong>Strategic Evaluation Function:</strong>
            <ul>
                <li><strong>Weighted Board Positions:</strong> Prioritized corner and edge control with a 12x12 positional weight matrix.</li>
                <li><strong>Mobility Scoring:</strong> Favored moves increasing player mobility while restricting opponents.</li>
                <li><strong>Piece Differential:</strong> Balanced piece count with bonuses for late-game stability.</li>
            </ul>
        </li>
    </ul>

    <h3>Technologies Used</h3>
    <ul>
        <li><strong>Algorithm:</strong> Minimax with alpha-beta pruning, iterative deepening (time-permitting).</li>
        <li><strong>Language:</strong> Python</li>
    </ul>

    <h3>Key Achievements</h3>
    <ul>
        <li>Outperformed reference agents by <strong>90%</strong> in timed matches through efficient pruning and adaptive depth selection.</li>
        <li>Reduced average move computation time by <strong>30%</strong> using positional heuristics and mobility-based evaluation.</li>
    </ul>

    <h3>Challenges & Solutions</h3>
    <ul>
        <li><strong>Large Board Complexity:</strong> Addressed 12x12 board size with optimized move generation and caching of legal moves.</li>
        <li><strong>Time Constraints:</strong> Implemented time-aware depth selection to avoid timeout losses in late-game scenarios.</li>
        <li><strong>Evaluation Tuning:</strong> Calibrated positional weights using iterative testing against reference agents.</li>
    </ul>

    `,
    },

    4: {
      name: `<h5> Transfer Learning for Image Classification </h5>
          <br>`,
      images: ["assets/images/multi-class-ml.png"],
      description: `
          <h3>Project Overview</h3>
        <p>Developed a multi-class image classification system to distinguish between six scenes (buildings, forest, glacier, mountain, sea, and street) using <strong>transfer learning</strong> with pre-trained models (<strong>ResNet50, ResNet101, EfficientNetB0, and VGG16</strong>). Achieved <strong>91.83% test accuracy</strong> with EfficientNetB0, outperforming other models, and demonstrated exceptional class separation with AUC scores exceeding <strong>0.99</strong> across all architectures.</p>
          <br>

        <h3>Understanding Transfer Learning</h3>
        <p>When dealing with relatively small image datasets, deep networks may not perform well due to insufficient training data. <strong>Transfer learning</strong> helps overcome this by leveraging deep learning models trained on large datasets like <strong>ImageNet</strong> as feature extractors.</p>
        <p>In this approach, we <strong>remove the last few layers</strong> of the pre-trained network and use the response of the previous layer as a feature vector for the new dataset. We then train a new fully connected layer while keeping the earlier layers frozen, ensuring efficient feature extraction and generalization.</p>
          <br>

        <h3>Key Features</h3>
        <ul>
            <li><strong>Transfer Learning:</strong> Utilized <strong>ResNet50, ResNet101, EfficientNetB0, and VGG16</strong> as feature extractors, freezing all layers except the final fully connected layer.</li>
            <li><strong>Data Augmentation:</strong> Applied <strong>random cropping, zooming, rotation, flipping, and contrast adjustments</strong> to enhance dataset diversity (20% validation split).</li>
            <li><strong>Model Architecture:</strong> 
                <ul>
                    <li>Added <strong>Global Average Pooling, Batch Normalization, and Dropout (20%)</strong> layers for regularization.</li>
                    <li>Used <strong>ReLU activation</strong> and <strong>softmax</strong> for multi-class classification.</li>
                </ul>
            </li>
            <li><strong>Training:</strong> Trained for <strong>60 epochs</strong> with <strong>early stopping</strong>, using the <strong>Adam optimizer</strong> and <strong>categorical cross-entropy loss</strong>.</li>
            <li><strong>Evaluation:</strong> All models achieved <strong>>90% test accuracy</strong>, with EfficientNetB0 leading at <strong>91.83%</strong>. 
            <br>
            <br>
            <strong>Key metrics:</strong>
            <br>
                <table>
                    <tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>AUC</th></tr>
                    <tr><td>ResNet50</td><td>90.70%</td><td>90.68%</td><td>0.9931</td></tr>
                    <tr><td>ResNet101</td><td>91.70%</td><td>91.66%</td><td>0.9936</td></tr>
                    <tr><td>EfficientNetB0</td><td>91.83%</td><td>91.81%</td><td>0.9921</td></tr>
                    <tr><td>VGG16</td><td>90.63%</td><td>90.59%</td><td>0.9910</td></tr>
                </table>
            </li>
        </ul>

        <h3>Technologies Used</h3>
        <ul>
            <li><strong>Frameworks:</strong> TensorFlow, Keras</li>
            <li><strong>Pre-Trained Models:</strong> ResNet50, ResNet101, EfficientNetB0, VGG16</li>
            <li><strong>Optimization:</strong> L2 regularization, Batch Normalization, Dropout</li>
            <li><strong>Evaluation Metrics:</strong> Precision, Recall, F1 Score, AUC (all >90%)</li>
        </ul>

        <h3>Key Achievements</h3>
        <ul>
            <li><strong>State-of-the-art performance:</strong> EfficientNetB0 achieved <strong>91.83% test accuracy</strong> and <strong>0.992 AUC</strong>, outperforming deeper models like ResNet101.</li>
            <li><strong>Consistent results:</strong> All architectures exceeded <strong>90% accuracy</strong>, demonstrating the reliability of transfer learning for small datasets.</li>
        </ul>

        <h3>Challenges & Solutions</h3>
        <ul>
            <li><strong>Small Dataset:</strong> Addressed via aggressive data augmentation and frozen feature extraction layers.</li>
            <li><strong>Model Selection:</strong> EfficientNetB0 outperformed larger models, likely due to its parameter-efficient architecture.</li>
            <li><strong>Compute Constraints:</strong> Early stopping after 60 epochs balanced performance with resource efficiency.</li>
        </ul>

        <h3>Conclusion & Impact</h3>
        <p>The project demonstrates that transfer learning with modern architectures like EfficientNetB0 enables <strong>high-accuracy image classification (>90%)</strong> even on small datasets. The framework can be extended to other domains like medical imaging or satellite analysis.</p>
       
        `,
    },
    5: {
      name: `<h5> Performance Analysis of Diffusion Model for Cloud Removal from Satellite Images </h5>    
              <br>`,
      images: ["assets/images/cloud.png"],
      description: `
              <h3>Project Overview</h3>
<p>Developed a deep learning solution to reconstruct cloud-obscured regions in satellite imagery using <strong>Denoising Diffusion Probabilistic Models (DDPM)</strong>. The project addressed challenges in remote sensing by restoring missing data caused by cloud cover, achieving <strong>PSNR (Peak Signal-to-Noise Ratio): 30.25</strong> and <strong>SSIM (Structural Similarity Index): 0.9153</strong>, comparable to state-of-the-art methods. Introduced a novel application of diffusion models for satellite image inpainting, eliminating dependency on paired datasets.</p>
<br>

<h3>Understanding the Core Concept</h3>
<p>Diffusion models iteratively add and remove noise from data. Unlike traditional methods like <strong>GANs (Generative Adversarial Networks)</strong> or <strong>CNNs (Convolutional Neural Networks)</strong>, DDPMs generate images by reversing a gradual noising process, ensuring structural coherence. This project adapted DDPMs for <strong>mask-guided inpainting</strong>, where cloud-covered regions (masked pixels) were regenerated using context from surrounding areas.</p>
<br>

<h3>Key Features</h3>
<ul>
    <li><strong>Mask-Based Inpainting:</strong> Generated cloud-free regions using unconditional DDPMs conditioned on masked inputs, preserving spatial consistency without auxiliary data.</li>
    <li><strong>Model Fine-Tuning:</strong> Adapted a pre-trained DDPM (Denoising Diffusion Probabilistic Model) to satellite imagery, reducing training time by 40%.</li>
    <li><strong>Sampling Optimization:</strong> Implemented resampling during reverse diffusion to harmonize generated regions with existing cloud-free areas.</li>
    <li><strong>Deployment:</strong> Published the model on <strong>Hugging Face</strong> and open-sourced code on <strong>GitHub</strong> for community use.</li>
</ul>
<br>

<h3>Technologies Used</h3>
<ul>
<li><strong>Language:</strong> Python</li>
    <li><strong>Frameworks:</strong> PyTorch, Hugging Face Diffusers</li>    
    <li><strong>Libraries:</strong> NumPy, Matplotlib</li>
    <li><strong>Evaluation Metrics:</strong> SSIM, PSNR, MSE (Mean Squared Error)</li>
</ul>
<br>

<h3>Key Achievements</h3>
<ul>
    <li>Achieved <strong>SSIM: 0.9153</strong> and <strong>PSNR: 30.25</strong>, outperforming GAN/CNN-based methods for small-to-moderate cloud coverage.</li>
    <li>Reduced training overhead by fine-tuning a pre-trained DDPM (Denoising Diffusion Probabilistic Models) on satellite data.</li>
    <li>Open-sourced code and model, enabling broader research in diffusion-based remote sensing.</li>
</ul>
<br>

<h3>Impact</h3>

<p>Demonstrated the viability of diffusion models for remote sensing applications, enabling accurate cloud removal without paired data. The work supports critical use cases like disaster response, agriculture, and environmental monitoring. The open-source release fosters collaboration in generative AI for geospatial analysis.</p>
                `,
    },
    6: {
      name: `<h5>Brain Tumor Detection using Convolutional Neural Networks (CNNs)</h5>
                    <br>`,
      images: ["assets/images/brain.png"],
      description: `
                    <h3>Project Overview</h3>
                    <p>Developed a CNN-based deep learning model to classify MRI brain scans as tumorous or healthy, achieving <strong>89% accuracy</strong> on test data. The system assists radiologists in early diagnosis by automating preliminary tumor detection, reducing manual analysis time.</p>
                    <br>
            
                    <h3>Understanding the Core Concept</h3>
                    <p>Convolutional Neural Networks (CNNs) excel at extracting spatial patterns from medical images. This model processes grayscale MRI scans through a series of convolution, pooling, and dropout layers to learn tumor signatures.</p>
                    <br>
            
                    <h3>Key Features</h3>
                    <ul>
                        <li><strong>Custom CNN Architecture:</strong> 4 convolutional layers with Leaky ReLU activations, batch normalization, and dropout (25-50%) for robust feature learning.</li>
                        <li><strong>Data Augmentation:</strong> Generated synthetic training samples using TensorFlow's <code>image_data_generator</code> to address class imbalance.</li>
                        <li><strong>Validation Pipeline:</strong> Split data into training (80%), validation (20%), and test sets for reliable performance metrics.</li>
                        <li><strong>Confusion Matrix Analysis:</strong> Achieved 99 true negatives and 79 true positives, identifying opportunities to reduce false negatives.</li>
                    </ul>
                    <br>
            
                    <h3>Technologies Used</h3>
                    <ul>
                        <li><strong>Languages:</strong> R </li>
                        <li><strong>Frameworks:</strong> Keras, TensorFlow</li>
                    </ul>
                    <br>
            
                    <h3>Key Achievements</h3>
                    <ul>
                        <li>Built a lightweight model with <strong>89% test accuracy</strong>.</li>
                        <li>Processed high-resolution MRI scans (124x124 pixels) efficiently using grayscale conversion and batch normalization.</li>
                        <li>Leveraged R's <code>magick</code> library for exploratory image analysis and preprocessing.</li>
                    </ul>
                    <br>
            
                    <h3>Impact</h3>
                    <p>Demonstrated the viability of CNNs for medical imaging tasks, providing a foundation for deploying AI-assisted diagnostic tools in healthcare. The modular architecture can be adapted to other classification tasks like detecting hemorrhages or anomalies in X-rays.</p>
                `,
    },
    7: {
      name: `<h5>Android Stock Trading App</h5><br>`,
      images: ["assets/images/stockapp.png"],
      description: `
                  <h3>Project Overview</h3>
                  <p>Developed a dynamic Android application for virtual stock trading, leveraging real-time data from <strong>Finnhub APIs</strong>. The app enables users to explore stocks, execute simulated trades, and manage portfolios with intuitive features like favorites tracking, interactive charts, and news aggregation. This project emphasizes on seamless integration of frontend and backend technologies.</p>
                  <br>
                  <h3>Key Features</h3>
                  <ul>
                      <li><strong>Real-Time Portfolio Management:</strong> Track cash balance, net worth, and stock performance with auto-updating values every 15 seconds.</li>
                      <li><strong>Favorites System:</strong> Swipe-to-delete and drag-and-reorder functionality for personalized watchlists.</li>
                      <li><strong>Interactive Charts:</strong> Visualize hourly and historical trends using HighCharts embedded in WebViews.</li>
                      <li><strong>Trade Execution:</strong> Virtual buy/sell transactions with error handling for invalid inputs (e.g., insufficient funds, non-positive shares).</li>
                      <li><strong>News Integration:</strong> Curated articles with clickable links to sources, Twitter, and Facebook sharing options.</li>
                      <li><strong>Autocomplete Search:</strong> Finnhub API-powered symbol suggestions for efficient navigation.</li>
                  </ul>
          
                  <h3>Technical Highlights</h3>
                  <ul>
                      <li><strong>Backend Integration:</strong> Node.js server to proxy Finnhub API requests, ensuring secure and efficient data retrieval.</li>
                      <li><strong>Data Persistence:</strong> MongoDB for storing user portfolios, favorites, and cash balances.</li>
                      <li><strong>Asynchronous Operations:</strong> Volley for non-blocking HTTP requests; Glide for image loading and caching.</li>
                      <li><strong>UI/UX:</strong> Material Design-compliant layouts using RecyclerView (SectionedAdapter), ConstraintLayout, and ViewPager2 with TabLayout.</li>
                      <li><strong>Error Handling:</strong> Graceful degradation for API failures, input validation, and crash prevention.</li>
                  </ul>
          
                  <h3>Tools & Technologies</h3>
                  <ul>
                      <li><strong>Languages:</strong> Java, JSON</li>
                      <li><strong>Frameworks:</strong> Android SDK, Node.js</li>
                      <li><strong>APIs:</strong> Finnhub (Company Profile, Quotes, News, Social Sentiments)</li>
                      <li><strong>Libraries:</strong> Volley, Glide, HighCharts, SectionedRecyclerViewAdapter</li>
                      <li><strong>Database:</strong> MongoDB</li>
                      <li><strong>Cloud:</strong> AWS (Deployment)</li>
                      <li><strong>IDE:</strong> Android Studio</li>
                  </ul>
          
                  <h3>Key Takeaways</h3>
                  <ul>
                      <li>Mastered end-to-end Android development, from UI design to backend integration.</li>
                      <li>Enhanced problem-solving skills through complex features like swipe gestures and real-time data synchronization.</li>
                      <li>Strengthened understanding of RESTful APIs, asynchronous programming, and Material Design principles.</li>
                      <li>Demonstrated ability to deliver a polished, user-centric application under technical constraints.</li>
                  </ul>
                          <p><strong>Note:</strong> The source code for this project is in a <strong>private GitHub repository</strong>. If you would like to <strong>view</strong> the code, please contact me via <a href="mailto:parthdpraja@gmail.com">email</a> or LinkedIn.</p>

              `,
    },

    // Add more projects as needed
  };

  detailsBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      const projectId = this.getAttribute("data-project");
      const project = projects[projectId];

      if (project) {
        // Update modal content
        document.getElementById("modal-project-name").innerHTML = project.name;
        document.getElementById("modal-project-images").innerHTML =
          project.images
            .map((img) => `<img src="${img}" alt="${project.name} Image">`)
            .join("");
        document.getElementById("modal-project-description").innerHTML =
          project.description;

        // Show the modal
        modal.style.display = "block";
      }
    });
  });

  closeBtn.addEventListener("click", function () {
    modal.style.display = "none";
  });

  window.addEventListener("click", function (event) {
    if (event.target === modal) {
      modal.style.display = "none";
    }
  });
});
