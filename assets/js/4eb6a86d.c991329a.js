"use strict";(self.webpackChunkdbt_osmosis=self.webpackChunkdbt_osmosis||[]).push([[739],{7032:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>t,contentTitle:()=>d,default:()=>h,frontMatter:()=>i,metadata:()=>o,toc:()=>a});const o=JSON.parse('{"id":"tutorial-yaml/context","title":"Context Variables","description":"dbt-osmosis provides three primary variables\u2014, {node}, and {parent}\u2014that can be referenced in your +dbt-osmosis: path configurations. These variables let you build powerful and dynamic rules for where your YAML files should live, all while staying DRY (don\u2019t repeat yourself).","source":"@site/docs/tutorial-yaml/context.md","sourceDirName":"tutorial-yaml","slug":"/tutorial-yaml/context","permalink":"/dbt-osmosis/docs/tutorial-yaml/context","draft":false,"unlisted":false,"editUrl":"https://github.com/z3z1ma/dbt-osmosis/tree/main/docs/docs/tutorial-yaml/context.md","tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2},"sidebar":"tutorialSidebar","previous":{"title":"Configuration","permalink":"/dbt-osmosis/docs/tutorial-yaml/configuration"},"next":{"title":"Inheritance","permalink":"/dbt-osmosis/docs/tutorial-yaml/inheritance"}}');var l=s(4848),r=s(8453);const i={sidebar_position:2},d="Context Variables",t={},a=[{value:"<code>{model}</code>",id:"model",level:2},{value:"Why Use <code>{model}</code>?",id:"why-use-model",level:3},{value:"<code>{node}</code>",id:"node",level:2},{value:"Creative Use Cases",id:"creative-use-cases",level:3},{value:"<code>{parent}</code>",id:"parent",level:2},{value:"Why Use <code>{parent}</code>?",id:"why-use-parent",level:3},{value:"Putting It All Together",id:"putting-it-all-together",level:2}];function c(e){const n={code:"code",em:"em",h1:"h1",h2:"h2",h3:"h3",header:"header",hr:"hr",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(n.header,{children:(0,l.jsx)(n.h1,{id:"context-variables",children:"Context Variables"})}),"\n",(0,l.jsxs)(n.p,{children:["dbt-osmosis provides three primary variables\u2014",(0,l.jsx)(n.code,{children:"{model}"}),", ",(0,l.jsx)(n.code,{children:"{node}"}),", and ",(0,l.jsx)(n.code,{children:"{parent}"}),"\u2014that can be referenced in your ",(0,l.jsx)(n.code,{children:"+dbt-osmosis:"})," path configurations. These variables let you build ",(0,l.jsx)(n.strong,{children:"powerful"})," and ",(0,l.jsx)(n.strong,{children:"dynamic"})," rules for where your YAML files should live, all while staying ",(0,l.jsx)(n.strong,{children:"DRY"})," (don\u2019t repeat yourself)."]}),"\n",(0,l.jsx)(n.h2,{id:"model",children:(0,l.jsx)(n.code,{children:"{model}"})}),"\n",(0,l.jsxs)(n.p,{children:["This variable expands to the ",(0,l.jsx)(n.strong,{children:"model name"})," being processed. If your model file is named ",(0,l.jsx)(n.code,{children:"stg_marketo__leads.sql"}),", ",(0,l.jsx)(n.code,{children:"{model}"})," will be ",(0,l.jsx)(n.code,{children:"stg_marketo__leads"}),"."]}),"\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Usage Example"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",metastring:'title="dbt_project.yml"',children:'models:\n  your_project_name:\n    # A default configuration that places each model\'s docs in a file named after the model,\n    # prefixed with an underscore\n    +dbt-osmosis: "_{model}.yml"\n\n    intermediate:\n      # Overrides the default in the \'intermediate\' folder:\n      # places YAMLs in a nested folder path, grouping them in "some/deeply/nested/path/"\n      +dbt-osmosis: "some/deeply/nested/path/{model}.yml"\n'})}),"\n",(0,l.jsxs)(n.h3,{id:"why-use-model",children:["Why Use ",(0,l.jsx)(n.code,{children:"{model}"}),"?"]}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"One-file-per-model"})," strategy: ",(0,l.jsx)(n.code,{children:"_{model}.yml"})," => ",(0,l.jsx)(n.code,{children:"_stg_marketo__leads.yml"})]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"Direct mapping"})," of model name to YAML file, making it easy to find"]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"Simple"})," approach when you want each model\u2019s metadata stored separately"]}),"\n"]}),"\n",(0,l.jsx)(n.h2,{id:"node",children:(0,l.jsx)(n.code,{children:"{node}"})}),"\n",(0,l.jsxs)(n.p,{children:[(0,l.jsx)(n.code,{children:"{node}"})," is a ",(0,l.jsx)(n.strong,{children:"powerful"})," placeholder giving you the entire node object as it appears in the manifest. This object includes details like:"]}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.fqn"}),": A list describing the folder structure (e.g., ",(0,l.jsx)(n.code,{children:'["my_project", "staging", "salesforce", "contacts"]'}),")"]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.resource_type"}),": ",(0,l.jsx)(n.code,{children:"model"}),", ",(0,l.jsx)(n.code,{children:"source"}),", or ",(0,l.jsx)(n.code,{children:"seed"})]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.language"}),": Typically ",(0,l.jsx)(n.code,{children:'"sql"'})]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.config[materialized]"}),": The model\u2019s materialization (e.g. ",(0,l.jsx)(n.code,{children:'"table"'}),", ",(0,l.jsx)(n.code,{children:'"view"'}),", ",(0,l.jsx)(n.code,{children:'"incremental"'}),")"]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.tags"}),": A list of tags you assigned in your model config"]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.code,{children:"node.name"}),": The name of the node (same as ",(0,l.jsx)(n.code,{children:"{model}"}),", but you get it as ",(0,l.jsx)(n.code,{children:"node.name"}),")"]}),"\n"]}),"\n",(0,l.jsxs)(n.p,{children:["With this variable, you can reference ",(0,l.jsx)(n.strong,{children:"any"})," node attribute directly in your file path."]}),"\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Usage Example"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",metastring:'title="dbt_project.yml"',children:'models:\n  jaffle_shop:\n    # We have a default config somewhere higher up. Now we override for intermediate or marts subfolders.\n\n    intermediate:\n      # advanced usage: use a combination of node.fqn, resource_type, language, and name\n      +dbt-osmosis: "node.fqn[-2]/{node.resource_type}_{node.language}/{node.name}.yml"\n\n    marts:\n      # more advanced: nest YAML by materialization, then by the first tag.\n      +dbt-osmosis: "node.config[materialized]/node.tags[0]/schema.yml"\n'})}),"\n",(0,l.jsx)(n.h3,{id:"creative-use-cases",children:"Creative Use Cases"}),"\n",(0,l.jsxs)(n.ol,{children:["\n",(0,l.jsxs)(n.li,{children:["\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Sort YAML by materialization"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",children:'+dbt-osmosis: "{node.config[materialized]}/{model}.yml"\n'})}),"\n",(0,l.jsxs)(n.p,{children:["If your model is a ",(0,l.jsx)(n.code,{children:"table"}),", the file path might become ",(0,l.jsx)(n.code,{children:"table/stg_customers.yml"}),"."]}),"\n"]}),"\n",(0,l.jsxs)(n.li,{children:["\n",(0,l.jsxs)(n.p,{children:[(0,l.jsx)(n.strong,{children:"Sort YAML by a specific tag"})," (you could use ",(0,l.jsx)(n.code,{children:"meta"})," as well)"]}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",children:'+dbt-osmosis: "node.tags[0]/{model}.yml"\n'})}),"\n",(0,l.jsxs)(n.p,{children:["If the first tag is ",(0,l.jsx)(n.code,{children:"finance"}),", you\u2019d get ",(0,l.jsx)(n.code,{children:"finance/my_model.yml"}),"."]}),"\n"]}),"\n",(0,l.jsxs)(n.li,{children:["\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Split by subfolders"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",children:'+dbt-osmosis: "{node.fqn[-2]}/{model}.yml"\n'})}),"\n",(0,l.jsx)(n.p,{children:"This references the \u201csecond-last\u201d element in your FQN array, often the subfolder name."}),"\n"]}),"\n",(0,l.jsxs)(n.li,{children:["\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Multi-level grouping"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",children:'+dbt-osmosis: "{node.resource_type}/{node.config[materialized]}/{node.name}.yml"\n'})}),"\n",(0,l.jsxs)(n.p,{children:["Group by whether it\u2019s a model, source, or seed, ",(0,l.jsx)(n.em,{children:"and"})," by its materialization."]}),"\n"]}),"\n"]}),"\n",(0,l.jsxs)(n.p,{children:["In short, ",(0,l.jsx)(n.code,{children:"{node}"})," is extremely flexible if you want to tailor your YAML file structure to reflect deeper aspects of the model\u2019s metadata."]}),"\n",(0,l.jsx)(n.h2,{id:"parent",children:(0,l.jsx)(n.code,{children:"{parent}"})}),"\n",(0,l.jsxs)(n.p,{children:["This variable represents the ",(0,l.jsx)(n.strong,{children:"immediate parent directory"})," of the ",(0,l.jsx)(n.strong,{children:"YAML file"})," that\u2019s being generated, which typically aligns with the folder containing the ",(0,l.jsx)(n.code,{children:".sql"})," model file. For example, if you have:"]}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{children:"models/\n  staging/\n    salesforce/\n      opportunities.sql\n"})}),"\n",(0,l.jsxs)(n.p,{children:["The ",(0,l.jsx)(n.code,{children:"{parent}"})," for ",(0,l.jsx)(n.code,{children:"opportunities.sql"})," is ",(0,l.jsx)(n.code,{children:"salesforce"}),". Thus if you do ",(0,l.jsx)(n.code,{children:'+dbt-osmosis: "{parent}.yml"'}),", you\u2019ll end up with a single ",(0,l.jsx)(n.code,{children:"salesforce.yml"})," in the ",(0,l.jsx)(n.code,{children:"staging/salesforce/"})," folder (lumping all models in that folder together)."]}),"\n",(0,l.jsx)(n.p,{children:(0,l.jsx)(n.strong,{children:"Usage Example"})}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",metastring:'title="dbt_project.yml"',children:'models:\n  jaffle_shop:\n    staging:\n      # So models in staging/salesforce => salesforce.yml\n      # models in staging/marketo => marketo.yml\n      # etc.\n      +dbt-osmosis: "{parent}.yml"\n'})}),"\n",(0,l.jsxs)(n.h3,{id:"why-use-parent",children:["Why Use ",(0,l.jsx)(n.code,{children:"{parent}"}),"?"]}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"Consolidated"})," YAML: All models in a given folder share a single YAML. For example, ",(0,l.jsx)(n.code,{children:"staging/salesforce/salesforce.yml"})," for 2\u20133 \u201csalesforce\u201d models."]}),"\n",(0,l.jsxs)(n.li,{children:["Great for ",(0,l.jsx)(n.strong,{children:"folder-based"})," org structures\u2014like ",(0,l.jsx)(n.code,{children:"staging/facebook_ads"}),", ",(0,l.jsx)(n.code,{children:"staging/google_ads"}),"\u2014and you want a single file for each source\u2019s staging models."]}),"\n"]}),"\n",(0,l.jsx)(n.hr,{}),"\n",(0,l.jsx)(n.h2,{id:"putting-it-all-together",children:"Putting It All Together"}),"\n",(0,l.jsxs)(n.p,{children:["You can mix and match these variables for ",(0,l.jsx)(n.strong,{children:"fine-grained"})," control. Here\u2019s a complex example that merges all:"]}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-yaml",children:'models:\n  my_project:\n    super_warehouse:\n      +dbt-osmosis: "{parent}/{node.config[materialized]}/{node.tags[0]}_{model}.yml"\n'})}),"\n",(0,l.jsxs)(n.ol,{children:["\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:(0,l.jsx)(n.code,{children:"{parent}"})})," => Name of the immediate subfolder under ",(0,l.jsx)(n.code,{children:"super_warehouse"}),"."]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:(0,l.jsx)(n.code,{children:"{node.config[materialized]}"})})," => Another subfolder named after the model\u2019s materialization."]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:(0,l.jsx)(n.code,{children:"{node.tags[0]}"})})," => A prefix in the filename, e.g. ",(0,l.jsx)(n.code,{children:"marketing_"})," or ",(0,l.jsx)(n.code,{children:"analytics_"}),"."]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:(0,l.jsx)(n.code,{children:"{model}"})})," => The actual model name for clarity."]}),"\n"]}),"\n",(0,l.jsxs)(n.p,{children:["So if you have a model ",(0,l.jsx)(n.code,{children:"super_warehouse/snapshots/payment_stats.sql"})," with ",(0,l.jsx)(n.code,{children:"materialized='table'"})," and a first tag of ",(0,l.jsx)(n.code,{children:"'billing'"}),", it might produce:"]}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{children:"super_warehouse/models/table/billing_payment_stats.yml\n"})}),"\n",(0,l.jsxs)(n.p,{children:["This approach ensures your YAML files reflect ",(0,l.jsx)(n.strong,{children:"both"})," how your code is organized (folder structure) ",(0,l.jsx)(n.strong,{children:"and"})," the model\u2019s metadata (materialization, tags, etc.), with minimal manual overhead."]}),"\n",(0,l.jsx)(n.hr,{}),"\n",(0,l.jsxs)(n.p,{children:[(0,l.jsx)(n.strong,{children:"In summary"}),", ",(0,l.jsx)(n.strong,{children:"context variables"})," are the backbone of dbt-osmosis\u2019s dynamic file routing system. With ",(0,l.jsx)(n.code,{children:"{model}"}),", ",(0,l.jsx)(n.code,{children:"{node}"}),", and ",(0,l.jsx)(n.code,{children:"{parent}"}),", you can define a wide range of file layout patterns and rely on dbt-osmosis to keep everything consistent. Whether you choose a single YAML per model, a single YAML per folder, or a more exotic arrangement that depends on tags, materializations, or your node\u2019s FQN, dbt-osmosis will automatically ",(0,l.jsx)(n.strong,{children:"organize"})," and ",(0,l.jsx)(n.strong,{children:"update"})," your YAMLs to match your declared config."]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,l.jsx)(n,{...e,children:(0,l.jsx)(c,{...e})}):c(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>i,x:()=>d});var o=s(6540);const l={},r=o.createContext(l);function i(e){const n=o.useContext(r);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function d(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(l):e.components||l:i(e.components),o.createElement(r.Provider,{value:n},e.children)}}}]);