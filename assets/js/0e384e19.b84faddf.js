"use strict";(self.webpackChunkdbt_osmosis=self.webpackChunkdbt_osmosis||[]).push([[976],{7879:(e,s,t)=>{t.r(s),t.d(s,{assets:()=>l,contentTitle:()=>d,default:()=>h,frontMatter:()=>r,metadata:()=>n,toc:()=>a});const n=JSON.parse('{"id":"intro","title":"dbt-osmosis Intro","description":"Let\'s discover dbt-osmosis in less than 5 minutes.","source":"@site/docs/intro.md","sourceDirName":".","slug":"/intro","permalink":"/dbt-osmosis/docs/intro","draft":false,"unlisted":false,"editUrl":"https://github.com/z3z1ma/dbt-osmosis/tree/main/docs/docs/intro.md","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","next":{"title":"dbt-osmosis - Basics","permalink":"/dbt-osmosis/docs/category/dbt-osmosis---basics"}}');var o=t(4848),i=t(8453);const r={sidebar_position:1},d="dbt-osmosis Intro",l={},a=[{value:"Getting Started",id:"getting-started",level:2},{value:"What you&#39;ll need",id:"what-youll-need",level:3},{value:"Configure dbt-osmosis",id:"configure-dbt-osmosis",level:2},{value:"Run dbt-osmosis",id:"run-dbt-osmosis",level:2}];function c(e){const s={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(s.header,{children:(0,o.jsx)(s.h1,{id:"dbt-osmosis-intro",children:"dbt-osmosis Intro"})}),"\n",(0,o.jsxs)(s.p,{children:["Let's discover ",(0,o.jsx)(s.strong,{children:"dbt-osmosis"})," in less than 5 minutes."]}),"\n",(0,o.jsx)(s.h2,{id:"getting-started",children:"Getting Started"}),"\n",(0,o.jsxs)(s.p,{children:["Get started by ",(0,o.jsx)(s.strong,{children:"running dbt-osmosis"}),"."]}),"\n",(0,o.jsx)(s.h3,{id:"what-youll-need",children:"What you'll need"}),"\n",(0,o.jsxs)(s.ul,{children:["\n",(0,o.jsxs)(s.li,{children:[(0,o.jsx)(s.a,{href:"https://www.python.org/downloads/",children:"Python"})," (3.9+)"]}),"\n",(0,o.jsxs)(s.li,{children:[(0,o.jsx)(s.a,{href:"https://docs.getdbt.com/docs/core/installation",children:"dbt"})," (1.8.0+)"]}),"\n",(0,o.jsxs)(s.li,{children:["or ",(0,o.jsx)(s.a,{href:"https://docs.astral.sh/uv/getting-started/installation/#standalone-installer",children:"uv"})]}),"\n",(0,o.jsxs)(s.li,{children:["An existing dbt project (or you can play with it using ",(0,o.jsx)(s.a,{href:"https://github.com/dbt-labs/jaffle_shop_duckdb",children:"jaffle shop"}),")"]}),"\n"]}),"\n",(0,o.jsx)(s.h2,{id:"configure-dbt-osmosis",children:"Configure dbt-osmosis"}),"\n",(0,o.jsxs)(s.p,{children:["Add the following to your ",(0,o.jsx)(s.code,{children:"dbt_project.yml"})," file. This example configuration tells dbt-osmosis that for every model in your project, there should exist a YAML file in the same directory with the same name as the model prefixed with an underscore. For example, if you have a model named ",(0,o.jsx)(s.code,{children:"my_model"})," then there should exist a YAML file named ",(0,o.jsx)(s.code,{children:"_my_model.yml"})," in the same directory as the model. The configuration is extremely flexible and can be used to declaratively organize your YAML files in any way you want as you will see later."]}),"\n",(0,o.jsx)(s.pre,{children:(0,o.jsx)(s.code,{className:"language-yaml",metastring:'title="dbt_project.yml"',children:'models:\n  your_project_name:\n    +dbt-osmosis: "_{model}.yml"\nseeds:\n  your_project_name:\n    +dbt-osmosis: "_schema.yml"\n'})}),"\n",(0,o.jsx)(s.h2,{id:"run-dbt-osmosis",children:"Run dbt-osmosis"}),"\n",(0,o.jsx)(s.p,{children:"If using uv(x):"}),"\n",(0,o.jsx)(s.pre,{children:(0,o.jsx)(s.code,{className:"language-bash",children:"uvx --with='dbt-<adapter>==1.9.0' dbt-osmosis yaml refactor\n"})}),"\n",(0,o.jsx)(s.p,{children:"Or, if installed in your Python environment:"}),"\n",(0,o.jsx)(s.pre,{children:(0,o.jsx)(s.code,{className:"language-bash",children:"dbt-osmosis yaml refactor\n"})}),"\n",(0,o.jsxs)(s.p,{children:["Run this command from the root of your dbt project. Ensure your git repository is clean before running. Replace ",(0,o.jsx)(s.code,{children:"<adapter>"})," with the name of your dbt adapter (e.g. ",(0,o.jsx)(s.code,{children:"snowflake"}),", ",(0,o.jsx)(s.code,{children:"bigquery"}),", ",(0,o.jsx)(s.code,{children:"redshift"}),", ",(0,o.jsx)(s.code,{children:"postgres"}),", ",(0,o.jsx)(s.code,{children:"athena"}),", ",(0,o.jsx)(s.code,{children:"spark"}),", ",(0,o.jsx)(s.code,{children:"trino"}),", ",(0,o.jsx)(s.code,{children:"sqlite"}),", ",(0,o.jsx)(s.code,{children:"duckdb"}),", ",(0,o.jsx)(s.code,{children:"oracle"}),", ",(0,o.jsx)(s.code,{children:"sqlserver"}),")."]}),"\n",(0,o.jsx)(s.p,{children:"Watch the magic unfold. \u2728"})]})}function h(e={}){const{wrapper:s}={...(0,i.R)(),...e.components};return s?(0,o.jsx)(s,{...e,children:(0,o.jsx)(c,{...e})}):c(e)}},8453:(e,s,t)=>{t.d(s,{R:()=>r,x:()=>d});var n=t(6540);const o={},i=n.createContext(o);function r(e){const s=n.useContext(i);return n.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function d(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:r(e.components),n.createElement(i.Provider,{value:s},e.children)}}}]);