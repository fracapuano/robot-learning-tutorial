% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A style for AI2 pre-prints
% Author: jacobm@allenai.org
% Version: 1.1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hfstyle/hf}
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm, margin=1.9cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}
\RequirePackage{fancyhdr}  % custom headers/footers

% \DisableLigatures[f]{family=sf*} 

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[most]{tcolorbox}
\definecolor{ai2accent}{HTML}{407579}
% \definecolor{ai2accent}{HTML}{ff0000}
\definecolor{hfforeground}{HTML}{1C2B33}
\definecolor{hfbackground}{HTML}{ffffb7}
\definecolor{hfforegroundDark}{HTML}{0A2B35}
\definecolor{ai2pink}{HTML}{F0529C}
\definecolor{hfyellow}{HTML}{000000}


% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=ai2accent,
  citecolor=ai2accent,
  urlcolor=ai2accent,
  anchorcolor=ai2accent,
  menucolor=ai2accent,
  filecolor=ai2accent,
  % linktocpage=true,
  allcolors=ai2accent
}


\RequirePackage[noabbrev,nameinlink]{cleveref}
% Reapply hyperref settings after cleveref to ensure they stick
\AtBeginDocument{
  \hypersetup{
    allcolors=ai2accent,
    linkcolor=ai2accent,
    citecolor=ai2accent,
    urlcolor=ai2accent
  }
}


% change base color of text 
\AtBeginDocument{
  \color{hfforegroundDark}
  \pagecolor{white}
}


\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}


% Create a custom size that's exactly 1pt larger than normalsize
\newcommand{\slightlylarger}{%
  \fontsize{\dimexpr\f@size pt+1}{\dimexpr\f@size pt-0.2\baselineskip}\selectfont%
}

% Create a custom size that's exactly 1pt smaller than normalsize
\newcommand{\slightlysmaller}{%
  \fontsize{\dimexpr\f@size pt-1}{\dimexpr\f@size pt+0.2\baselineskip}\selectfont%
}

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}
% \titleformat*{\paragraph}{\bfseries}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\slightlylarger\sffamily\bfseries}
\titleformat*{\paragraph}{\slightlysmaller\sffamily\bfseries}

% make bolded text smaller to match with serif.
% \DeclareTextFontCommand{\textbf}{\bfseries\sffamily}
\DeclareTextFontCommand{\textbf}{\fontsize{9}{11}\selectfont\bfseries\sffamily}


\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\DeclareCaptionLabelSeparator{pipe}{ $\vert$ }% or $\vert$
\captionsetup{singlelinecheck=false,format=custom,labelsep=pipe,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=pipe,font=small}

% %%======== Header and Footer Content ========



% % HF custom fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % http://c.caignaert.free.fr/Install-ttf-Font.pdf

% Set the main font to Times New Roman
% \setmainfont{Manrope}
% Set sans-serif font to Manrope
\RequirePackage{ifxetex}
\usepackage{ifxetex}
\ifxetex
    \usepackage{fontspec}
    \setsansfont{Manrope}
\else
    \RequirePackage[T1]{fontenc}
    \usepackage[T1]{fontenc}
    \usepackage{hfstyle/manrope}
    \renewcommand{\sfdefault}{manrope}
\fi

% % \pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
% % \renewcommand{\sfdefault}{optimistic}

% \DeclareFontFamily{T1}{manrope}{}
% \DeclareFontShape{T1}{manrope}{m}{n}{<-> s * [0.88] ai2style/fonts/Manrope}{}
% \pdfmapline{+manrope < ai2style/fonts/Manrope.ttf <T1-WGL4.enc}
% \renewcommand{\sfdefault}{manrope}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imporve appearance of ToC

\RequirePackage{tocloft}

% Make sure the section titles in TOC are in Manrope
\renewcommand{\cfttoctitlefont}{\sffamily}
\renewcommand{\cftsecfont}{\sffamily}
\renewcommand{\cftsubsecfont}{\sffamily}
\renewcommand{\cftsubsubsecfont}{\sffamily}
\renewcommand{\cftparafont}{\sffamily}

% Make the page numbers in TOC use Manrope
\renewcommand{\cftsecpagefont}{\sffamily}
\renewcommand{\cftsubsecpagefont}{\sffamily}
\renewcommand{\cftsubsubsecpagefont}{\sffamily}
\renewcommand{\cftparapagefont}{\sffamily}

% % Adjust the spacing
% \setlength{\cftsecindent}{1.5em}
% \setlength{\cftsubsecindent}{3.0em}
% \setlength{\cftsubsubsecindent}{4.5em}
% \setlength{\cftparaindent}{6.0em}

% \setlength{\cftsecnumwidth}{2.5em}
% \setlength{\cftsubsecnumwidth}{3.5em}
% \setlength{\cftsubsubsecnumwidth}{4.5em}
% \setlength{\cftparanumwidth}{5.5em}

% % Adjust vertical spacing
% \setlength{\cftbeforesecskip}{5pt plus 1pt}
% \setlength{\cftbeforesubsecskip}{2pt plus 1pt}
% \setlength{\cftbeforesubsubsecskip}{2pt plus 1pt}

% % Style the dots
% \renewcommand{\cftsecdotsep}{1.5}
% \renewcommand{\cftsubsecdotsep}{1.5}
% \renewcommand{\cftsubsubsecdotsep}{1.5}
% \renewcommand{\cftparadotsep}{1.5}

% % Color for the dots (using the pre-defined ai2accent color)
% \renewcommand{\cftsecleader}{\color{ai2accent}\cftsecdotsep}
% \renewcommand{\cftsubsecleader}{\color{ai2accent}\cftsubsecdotsep}
% \renewcommand{\cftsubsubsecleader}{\color{ai2accent}\cftsubsubsecdotsep}
% \renewcommand{\cftparaleader}{\color{ai2accent}\cftparadotsep}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% KYLE TESTING SOME STUFF OUT
\newcommand\authorlistOne{}
\newcommand\authorlistTwo{}
\newcommand\authorlistThree{}
\newcommand\authorlistFour{}
\newcommand\authorlistFive{}


\newcommand\authorOne[2][]{\addtolist[#1]{#2}{\authorlistOne}{\authorformat}{\quad }}
\newcommand\authorTwo[2][]{\addtolist[#1]{#2}{\authorlistTwo}{\authorformat}{\quad }}
\newcommand\authorThree[2][]{\addtolist[#1]{#2}{\authorlistThree}{\authorformat}{\quad }}
\newcommand\authorFour[2][]{\addtolist[#1]{#2}{\authorlistFour}{\authorformat}{\quad }}
\newcommand\authorFive[2][]{\addtolist[#1]{#2}{\authorlistFive}{\authorformat}{\quad }}






% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect 
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
% \renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{\quad}}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily \bfseries #1} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

% switch to sans serif
\newcommand\sans[1][]{\sffamily #1}

\renewcommand{\abstract}[1]{\def\abstractlist{{\color{hfforeground} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}


\renewcommand{\title}[1]{\newcommand{\titlelist}{{\Huge\sffamily\bfseries\color{hfyellow} #1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\mainfigure}{}  % New command to store the figure
\newcommand{\maintable}{}   % New command to store the table

\newcommand{\setmainfigure}[1]{%
  \renewcommand{\mainfigure}{#1}%
}

\newcommand{\setmaintable}[1]{%
  \renewcommand{\maintable}{#1}%
}

\newcommand{\mymaketitle}{%
  \tcbset{enhanced,frame hidden}
  \tcbset{borderline={0.5mm}{0mm}{yellow!70}}
  %\tcbset{colframe=yellow!40!orange}
  %\tcbset{opacityframe=1.0}
  %\tcbset{boxrule=0.5mm,}
  \tcbset{left=0.5cm}
  \tcbset{right=0.5cm}
  \tcbset{top=0.5cm}
  \tcbset{bottom=0.5cm}
  \tcbset{arc=12pt}
  \tcbset{colback=hfbackground, opacityback=0.65}
  \tcbset{before skip=0pt}
  \tcbset{grow to left by=1.5pt}
  \tcbset{grow to right by=1.5pt}
  \tcbset{overlay={\node[
    anchor=north east,
    at= (frame.north east),
    xshift=-0.5cm,
    yshift=-0.25cm] {\includegraphics[width=3.57cm]{hfstyle/assets/huggingface.pdf}};}}
  
  \setlength{\parskip}{0cm}
  \nohyphens
  {
    \setstretch{1.618}
    \titlelist\par
  }
  \vskip 0.5cm
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0cm}
    {
      \authorlistOne \par
      \vspace{0.3cm}
      \authorlistTwo\par
      \vspace{0.3cm}
      \authorlistThree\par
      % \vspace{0.3cm}  
      % \authorlistFour\par
      % \vspace{0.3cm}  
      % \authorlistFive\par
      % \vspace{0.3cm}
      \affiliationlist\par
      \vspace{0.3cm}
      \contributionlist\par
      \vspace{0.5cm}
      \ifdefempty{\metadatalist}{}{\metadatalist}
    }
    
    % Insert the figure here
    \ifdefempty{\mainfigure}{}{%
      % \vspace{0.5cm}
      \mainfigure
      % \vspace{0.5cm}
    }
    
  \begin{tcolorbox}
    {\Large\sffamily\bfseries Abstract\par}
    \vspace{0.5cm}
    \abstractlist\par
  \end{tcolorbox}
  
  % Insert the table after the abstract
  \ifdefempty{\maintable}{}{%
    % \vspace{0.5cm}
    \maintable
    \vspace{0.5cm}
  }
  
  \tcbset{reset}
  \FloatBarrier
  \setlength{\parskip}{5.5\p@}
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.38cm
]
}%
\else%
\renewcommand{\maketitle}{
  \mymaketitle
}%
\fi