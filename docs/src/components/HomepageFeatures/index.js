import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Why Does this Exist',
    Svg: require('@site/static/img/dbt.svg').default,
    description: (
      <>
        With dbt, data teams work directly within the warehouse to produce
        trusted datasets for reporting, ML modeling, and operational workflows.
        Part of this workflow includes managing YAML files and metadata. Little
        things like column descriptions, tags, and tests can be a pain to
        maintain and tend not to be DRY.
      </>
    ),
  },
  {
    title: 'What Can we Automate',
    Svg: require('@site/static/img/yaml-icon.svg').default,
    description: (
      <>
        Most of the time, you don't <b>need</b> to write YAML by hand. dbt-osmosis
        automates the process of generating YAML files for you. You can focus on
        what matters. Source yaml files are generated from your database. Schema
        yaml files are generated from your dbt models with inherited metadata.
      </>
    ),
  },
  {
    title: 'Git Oriented',
    Svg: require('@site/static/img/github-icon.svg').default,
    description: (
      <>
        A single execution on your dbt project can save you hours of manual toil. dbt-osmosis
        is built to work with your existing dbt project. It can be run directly, as a pre-commit hook,
        or as a CI/CD step generating a PR. By leveraging git, we can safely execute
        file changes en masse with proper diffs and version control.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
